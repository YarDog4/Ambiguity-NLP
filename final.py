#Dual-channel text ensemble + rich image augmentation + quadrant crops + 80/20 split + Bayesian tuning
import os
import random
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn

from transformers import CLIPModel, CLIPProcessor

import optuna
from optuna.samplers import TPESampler

import time

############################## This is for measuring some data ##############################
def profile_existing_inference_cost(
    df,
    image_dict,
    model,
    processor,
    photo_weight=0.6,
    semantic_weight=0.4,
    temp=0.7,
    num_rows=50,
    warmup=5,
    seed=42,
    out_size=(224, 224),
):
    """
    Profiles computational cost EXACTLY as your current inference code does.

    Measures:
      1) Number of augmented views per image embedding (from your actual view pipeline)
      2) Latency for:
         - text embedding (dual-channel prompts)
         - image embedding (multi-view)
         - end-to-end choose_image (1 query with 10 candidate images)

    Prints an understandable summary (mean + p50/p90/p95).
    """

    def _device_of(m): 
        return next(m.parameters()).device

    def _sync(device):
        if device.type == "cuda":
            torch.cuda.synchronize()

    def _stats(seconds_list):
        arr = np.array(seconds_list, dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    def _ms(x): 
        return x * 1000.0

    # --- Count views using your *actual* view generation functions ---
    def _count_views(img):
        # Mirrors get_image_embedding()'s view construction in your current code
        views = []
        views.extend(generate_tta_views(img, num_random_augs=2, out_size=out_size))
        views.extend(multi_crops(img, out_size=out_size))
        views.extend(grid_patches(img, grid_size=3, out_size=out_size))
        views.extend(center_saliency_crops(img, out_size=out_size))
        views.extend(mid_quadrant_crops(img, out_size=out_size))
        return len(views)

    # --- Setup ---
    device = _device_of(model)
    model.eval()

    df = df.reset_index(drop=True)
    if len(df) == 0:
        print("[profile_existing_inference_cost] df is empty; nothing to profile.")
        return

    rng = np.random.default_rng(seed)
    n = min(num_rows, len(df))
    sample_idxs = rng.choice(len(df), size=n, replace=False)
    sample_df = df.iloc[sample_idxs]

    # Find one real image to count views on
    views_per_image = None
    for _, row in sample_df.iterrows():
        candidates = [row[f"image_{i}"] for i in range(10)]
        name = next((c for c in candidates if c in image_dict), None)
        if name is not None:
            views_per_image = _count_views(image_dict[name])
            break
    if views_per_image is None:
        views_per_image = 0  # if your dict doesn't contain the sampled filenames

    # --- Warmup (reduces first-run overhead) ---
    for _ in range(max(0, warmup)):
        r = sample_df.iloc[0]
        candidates = [r[f"image_{i}"] for i in range(10)]
        _ = get_dual_channel_text_embedding(
            target=r["target"],
            sentence=r["sentence"],
            processor=processor,
            model=model,
            photo_weight=photo_weight,
            semantic_weight=semantic_weight,
        )
        for c in candidates:
            if c in image_dict:
                _ = get_image_embedding(image_dict[c], processor=processor, model=model, temp=temp)
                break
        _ = choose_image(
            target=r["target"],
            sentence=r["sentence"],
            images=candidates,
            image_dict=image_dict,
            model=model,
            processor=processor,
            photo_weight=photo_weight,
            semantic_weight=semantic_weight,
            temp=temp,
            print_output=False,
        )
    _sync(device)

    # --- Timing ---
    text_times = []
    image_times = []
    query_times = []  # end-to-end choose_image

    for _, row in sample_df.iterrows():
        target = row["target"]
        sentence = row["sentence"]
        candidates = [row[f"image_{i}"] for i in range(10)]

        # 1) Text embedding time (matches your choose_image text path)
        _sync(device)
        t0 = time.perf_counter()
        _ = get_dual_channel_text_embedding(
            target=target,
            sentence=sentence,
            processor=processor,
            model=model,
            photo_weight=photo_weight,
            semantic_weight=semantic_weight,
        )
        _sync(device)
        t1 = time.perf_counter()
        text_times.append(t1 - t0)

        # 2) Image embedding time (matches your get_image_embedding for each candidate)
        for name in candidates:
            img = image_dict.get(name, None)
            if img is None:
                continue
            _sync(device)
            i0 = time.perf_counter()
            _ = get_image_embedding(img, processor=processor, model=model, temp=temp)
            _sync(device)
            i1 = time.perf_counter()
            image_times.append(i1 - i0)

        # 3) End-to-end choose_image time (the real inference call)
        _sync(device)
        q0 = time.perf_counter()
        _ = choose_image(
            target=target,
            sentence=sentence,
            images=candidates,
            image_dict=image_dict,
            model=model,
            processor=processor,
            photo_weight=photo_weight,
            semantic_weight=semantic_weight,
            temp=temp,
            print_output=False,
        )
        _sync(device)
        q1 = time.perf_counter()
        query_times.append(q1 - q0)

    # --- Summaries ---
    ts = _stats(text_times)
    ims = _stats(image_times)
    qs = _stats(query_times)

    # Estimate per-query image-embedding cost assuming 10 candidates
    # (This is a practical estimate readers like.)
    if not np.isnan(ims["mean"]):
        est_img_cost_per_query = ims["mean"] * 10.0
    else:
        est_img_cost_per_query = float("nan")

    print("\n==============================")
    print("COMPUTATIONAL COST (Matches Existing Code)")
    print("==============================")
    print(f"Device: {device}")
    print(f"Profiled queries: {n}")
    print(f"Candidates per query: 10")
    print(f"Augmented views per image embedding: {views_per_image}")
    print("")
    print("Text embedding latency (dual-channel prompts):")
    print(f"  Mean: {_ms(ts['mean']):.2f} ms | P50: {_ms(ts['p50']):.2f} | P90: {_ms(ts['p90']):.2f} | P95: {_ms(ts['p95']):.2f}")
    print("")
    print("Image embedding latency (multi-view CLIP per candidate image):")
    print(f"  Mean: {_ms(ims['mean']):.2f} ms/image | P50: {_ms(ims['p50']):.2f} | P90: {_ms(ims['p90']):.2f} | P95: {_ms(ims['p95']):.2f}")
    print(f"  Estimated image-embedding cost per query (10 candidates): {_ms(est_img_cost_per_query):.2f} ms/query")
    print("")
    print("End-to-end inference latency (choose_image on 10 candidates):")
    print(f"  Mean: {_ms(qs['mean']):.2f} ms/query | P50: {_ms(qs['p50']):.2f} | P90: {_ms(qs['p90']):.2f} | P95: {_ms(qs['p95']):.2f}")
    print("==============================\n")


############################## Load in the SemEval data ##############################
def load_data(file_path, train_val="trial", target_size=(384, 384), use_cache=True):
    """
    Load SemEval data and images with optional disk caching.

    Returns:
        data (DataFrame): columns [target, sentence, image_0-9, label]
        image_dict (dict): {filename -> PIL.Image}
    """
    # Train/trial/test set directory
    path = os.path.join(file_path, train_val + "_v1")

    # Cache file path
    cache_file = os.path.join(path, "image_cache.pkl")

    # Load in the data
    path_data = os.path.join(path, train_val + ".data.v1.txt")
    data = pd.read_csv(path_data, sep="\t", header=None)
    data.columns = ["target", "sentence"] + [f"image_{i}" for i in range(data.shape[1] - 2)]

    # Load in the labels
    path_labels = os.path.join(path, train_val + ".gold.v1.txt")
    with open(path_labels, "r", encoding="utf-8") as f:
        gold_labels = [line.strip() for line in f]
    data["label"] = gold_labels

    # Try to load cached images
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached images from {cache_file}...")
        with open(cache_file, "rb") as f:
            image_dict = pickle.load(f)
        print(f"Loaded {len(image_dict)} cached images")
        return data, image_dict

    # Load in the images (first time or if cache disabled)
    path_images = os.path.join(path, train_val + "_images_v1")
    image_dict = {}
    files = os.listdir(path_images)
    for filename in tqdm(files, total=len(files),
                         desc="Loading in the Images", unit="image"):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            try:
                img = Image.open(os.path.join(path_images, filename)).convert("RGB")
                img_resized = img.resize(target_size, resample=Image.BICUBIC)
                image_dict[filename] = img_resized
            except Exception:
                continue

    # Save to cache
    if use_cache:
        print(f"Saving images to cache: {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(image_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cached {len(image_dict)} images")

    return data, image_dict


############################## Text helpers (micro-prompts + dual-channel) ##############################
def extract_context_word(target: str, sentence: str) -> str:
    """
    Try to extract the 'context word' when the sentence looks like:
      - 'bank erosion'
      - 'internet router'
      - 'andromeda tree'
    If that fails, fall back to the full sentence as context.
    """
    if sentence is None:
        return ""

    s = str(sentence).strip()
    t = str(target).strip()

    if not s:
        return ""

    tokens = s.split()

    # Case 1: exactly two tokens, one is target → use the other as context
    if len(tokens) == 2:
        t_low = t.lower()
        tok0, tok1 = tokens[0], tokens[1]
        if tok0.lower() == t_low:
            return tok1
        if tok1.lower() == t_low:
            return tok0

    # Otherwise, just treat sentence as context phrase
    return s


def get_synonym(word: str) -> str | None:
    """
    Very lightweight WordNet synonym helper.
    Returns a single short synonym (or None if not available).
    """
    if word is None:
        return None

    w = str(word).strip()
    if not w:
        return None

    synsets = wn.synsets(w)
    if not synsets:
        return None

    lemmas = synsets[0].lemma_names()
    if not lemmas:
        return None

    return lemmas[0].replace("_", " ")


def build_photo_prompts(target: str, sentence: str) -> list[str]:
    """
    'Photo-style' prompts – closer to CLIP's pretraining distribution.
    """
    target = str(target).strip()
    sentence = "" if sentence is None else str(sentence).strip()

    context = extract_context_word(target, sentence).strip()
    base_context = context if context else sentence

    prompts: list[str] = []

    prompts.append(f"a photo of {target} {base_context}".strip())
    prompts.append(f"{target} {base_context}, realistic photo".strip())
    prompts.append(f"{target} near {base_context}, real world".strip())
    prompts.append(f"{target} with {base_context}, natural scene".strip())
    prompts.append(f"{target} appearing in a {base_context} environment".strip())
    prompts.append(f"{target} {base_context}, high quality photograph".strip())

    # synonym-boosted variant
    t_syn = get_synonym(target)
    c_syn = get_synonym(context) if context else None
    syn_target = t_syn if t_syn else target
    syn_context = c_syn if c_syn else base_context
    if syn_target and syn_context:
        prompts.append(f"a photo of {syn_target} {syn_context}".strip())

    return [p for p in prompts if p]


def build_semantic_prompts(target: str, sentence: str) -> list[str]:
    """
    'Semantic / caption-style' prompts that describe meaning and usage,
    still kept short for CLIP.
    """
    target = str(target).strip()
    sentence = "" if sentence is None else str(sentence).strip()

    context = extract_context_word(target, sentence).strip()
    base_context = context if context else sentence

    prompts: list[str] = []

    prompts.append(f"{target} related to {base_context}".strip())
    prompts.append(f"the concept of {target} in {base_context}".strip())
    prompts.append(f"{target} in the context of {base_context}".strip())
    prompts.append(f"visual sense of {target} in {base_context}".strip())
    prompts.append(f"illustration of {target} used with {base_context}".strip())

    # sentence-level rewrites
    if sentence:
        prompts.append(f"a visual depiction of: {sentence}".strip())
        prompts.append(f"illustration of the phrase: {sentence}".strip())
        prompts.append(f"{sentence} (image)".strip())
        prompts.append(f"{sentence}, realistic photography".strip())

    return [p for p in prompts if p]


def get_prompted_text_embedding(
    prompts: list[str],
    processor: CLIPProcessor,
    model: CLIPModel,
) -> torch.Tensor:
    """
    Encode a list of prompts with CLIP text encoder, then average + normalize.
    """
    model.eval()
    device = next(model.parameters()).device

    if len(prompts) == 0:
        # Fallback zero vector (should not really happen)
        d = model.config.projection_dim
        return torch.zeros(d, device=device)

    with torch.no_grad():
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)  # (P, d)
        feats = F.normalize(feats, p=2, dim=-1)
        mean_feat = feats.mean(dim=0)             # (d,)
        mean_feat = F.normalize(mean_feat, p=2, dim=-1)

    return mean_feat

def get_dual_channel_text_embedding(
    target: str,
    sentence: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    photo_weight: float = 0.6,
    semantic_weight: float = 0.4,
) -> torch.Tensor:
    """
    Dual-channel text embedding:
      - photo-style prompts channel
      - semantic/context prompts channel
    Then weighted combination + double normalization.
    """
    photo_prompts = build_photo_prompts(target, sentence)
    sem_prompts = build_semantic_prompts(target, sentence)

    photo_emb = get_prompted_text_embedding(photo_prompts, processor, model)
    sem_emb = get_prompted_text_embedding(sem_prompts, processor, model)

    combined = photo_weight * photo_emb + semantic_weight * sem_emb
    combined = F.normalize(combined, p=2, dim=-1)
    combined = F.normalize(combined, p=2, dim=-1)  # double normalization

    return combined  # (d,)


############################## Image Augmentations ##############################
def random_geometric_augment(img: Image.Image) -> Image.Image:
    """Mild random geometric transforms: flip, rotation, slight zoom/crop."""
    w, h = img.size
    out = img

    # Random horizontal flip (50% chance)
    if random.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)

    # Small random rotation
    angle = random.uniform(-7, 7)  # degrees
    out = out.rotate(angle, resample=Image.BICUBIC, expand=False)

    # Slight random crop (95–100% of size)
    scale = random.uniform(0.95, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w < w and new_h < h:
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        out = out.crop((left, top, left + new_w, top + new_h))
        out = out.resize((w, h), resample=Image.BICUBIC)

    return out

def random_photometric_augment(img: Image.Image) -> Image.Image:
    """Mild brightness/contrast/color/blur jitter."""
    out = img

    # Brightness
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(out)
        factor = random.uniform(0.95, 1.05)
        out = enhancer.enhance(factor)

    # Contrast
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(out)
        factor = random.uniform(0.95, 1.05)
        out = enhancer.enhance(factor)

    # Color
    if random.random() < 0.5:
        enhancer = ImageEnhance.Color(out)
        factor = random.uniform(0.95, 1.05)
        out = enhancer.enhance(factor)

    # Slight blur (very mild)
    if random.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=0.3))

    return out

def generate_tta_views(
    img: Image.Image,
    num_random_augs: int = 3,
    out_size=(224, 224),
) -> list[Image.Image]:
    """
    Strong TTA for CLIP:
    - original
    - horizontal flip
    - center crop
    - zoomed center
    - grayscale
    - a few random geometric+photometric augmentations
    """
    views = []
    w, h = img.size

    # Base resized view
    base = img.resize(out_size, resample=Image.BICUBIC)
    views.append(base)

    # Horizontal flip
    hflip = img.transpose(Image.FLIP_LEFT_RIGHT).resize(out_size, resample=Image.BICUBIC)
    views.append(hflip)

    # Center crop (80%)
    crop_scale = 0.8
    cw, ch = int(w * crop_scale), int(h * crop_scale)
    left = (w - cw) // 2
    top = (h - ch) // 2
    center_crop = img.crop((left, top, left + cw, top + ch))
    center_crop = center_crop.resize(out_size, resample=Image.BICUBIC)
    views.append(center_crop)

    # Zoomed center (60%)
    zoom_scale = 0.6
    zw, zh = int(w * zoom_scale), int(h * zoom_scale)
    zleft = (w - zw) // 2
    ztop = (h - zh) // 2
    zoom_crop = img.crop((zleft, ztop, zleft + zw, ztop + zh))
    zoom_crop = zoom_crop.resize(out_size, resample=Image.BICUBIC)
    views.append(zoom_crop)

    # Grayscale variant
    gray = img.convert("L").convert("RGB").resize(out_size, resample=Image.BICUBIC)
    views.append(gray)

    # A few random augmentations (geometric + photometric)
    for _ in range(num_random_augs):
        aug = random_geometric_augment(img)
        aug = random_photometric_augment(aug)
        aug = aug.resize(out_size, resample=Image.BICUBIC)
        views.append(aug)

    return views

def multi_crops(img: Image.Image, out_size=(224, 224)) -> list[Image.Image]:
    """Center + four corners + zoomed center (multi-crop)."""
    w, h = img.size
    crops = []

    # Main center crop (80%)
    crop_w, crop_h = int(w * 0.8), int(h * 0.8)
    crop_w = max(1, crop_w)
    crop_h = max(1, crop_h)

    # Center
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    center = img.crop((left, top, left + crop_w, top + crop_h))
    crops.append(center.resize(out_size, resample=Image.BICUBIC))

    # Top-left
    tl = img.crop((0, 0, crop_w, crop_h))
    crops.append(tl.resize(out_size, resample=Image.BICUBIC))

    # Top-right
    tr = img.crop((w - crop_w, 0, w, crop_h))
    crops.append(tr.resize(out_size, resample=Image.BICUBIC))

    # Bottom-left
    bl = img.crop((0, h - crop_h, crop_w, h))
    crops.append(bl.resize(out_size, resample=Image.BICUBIC))

    # Bottom-right
    br = img.crop((w - crop_w, h - crop_h, w, h))
    crops.append(br.resize(out_size, resample=Image.BICUBIC))

    # Zoomed-in center (60%)
    zoom_scale = 0.6
    z_w, z_h = int(w * zoom_scale), int(h * zoom_scale)
    z_left = (w - z_w) // 2
    z_top = (h - z_h) // 2
    zoom_center = img.crop((z_left, z_top, z_left + z_w, z_top + z_h))
    crops.append(zoom_center.resize(out_size, resample=Image.BICUBIC))

    return crops

def grid_patches(img: Image.Image, grid_size=3, out_size=(224, 224)) -> list[Image.Image]:
    """Split the image into a grid (e.g., 3x3) and return patches."""
    w, h = img.size
    patch_w = w // grid_size
    patch_h = h // grid_size

    patches = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            left = gx * patch_w
            top = gy * patch_h
            right = w if gx == grid_size - 1 else (left + patch_w)
            bottom = h if gy == grid_size - 1 else (top + patch_h)

            patch = img.crop((left, top, right, bottom))
            patches.append(patch.resize(out_size, resample=Image.BICUBIC))

    return patches

def center_saliency_crops(img: Image.Image, out_size=(224, 224)) -> list[Image.Image]:
    """
    Lightweight 'saliency-like' center crops: smaller crops around the center.
    Not true saliency, but emphasizes likely subject region.
    """
    w, h = img.size
    crops = []

    for scale in [0.7, 0.5]:
        cw, ch = int(w * scale), int(h * scale)
        left = (w - cw) // 2
        top = (h - ch) // 2
        crop = img.crop((left, top, left + cw, top + ch))
        crops.append(crop.resize(out_size, resample=Image.BICUBIC))

    return crops

def mid_quadrant_crops(img: Image.Image, out_size=(224, 224)) -> list[Image.Image]:
    """
    Mid-quadrant crops: centers of four quadrants between center and corners.
    Helps capture off-center important regions.
    """
    w, h = img.size
    crops = []

    cw, ch = w // 2, h // 2  # crop width/height ~ half
    # centers of quadrants
    centers = [
        (w // 4, h // 4),         # mid-top-left
        (3 * w // 4, h // 4),     # mid-top-right
        (w // 4, 3 * h // 4),     # mid-bottom-left
        (3 * w // 4, 3 * h // 4), # mid-bottom-right
    ]

    for cx, cy in centers:
        left = max(0, cx - cw // 2)
        top = max(0, cy - ch // 2)
        right = min(w, left + cw)
        bottom = min(h, top + ch)
        crop = img.crop((left, top, right, bottom))
        crops.append(crop.resize(out_size, resample=Image.BICUBIC))

    return crops

def get_image_embedding(
    img: Image.Image,
    processor: CLIPProcessor,
    model: CLIPModel,
    temp: float = 0.7,
    out_size=(224, 224),
) -> torch.Tensor:
    """
    Build a rich multi-view embedding:
    - Strong TTA views
    - Multi-crops (center + corners + zoomed)
    - Grid patches
    - Center-focused crops
    - Mid-quadrant crops
    Then:
    - Batch all views through CLIP
    - Average features
    - Normalize + temperature scaling
    """
    model.eval()
    device = next(model.parameters()).device

    # Collect all views
    views: list[Image.Image] = []
    views.extend(generate_tta_views(img, num_random_augs=2, out_size=out_size))
    views.extend(multi_crops(img, out_size=out_size))
    views.extend(grid_patches(img, grid_size=3, out_size=out_size))
    views.extend(center_saliency_crops(img, out_size=out_size))
    views.extend(mid_quadrant_crops(img, out_size=out_size))

    with torch.no_grad():
        inputs = processor(images=views, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # If on GPU, use autocast for speed
        use_amp = (device.type == "cuda") if isinstance(device, torch.device) else False
        if hasattr(torch.cuda, "amp"):
            ctx = torch.cuda.amp.autocast(enabled=use_amp)
        else:
            class DummyCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc_val, exc_tb): return False
            ctx = DummyCtx()

        with ctx:
            feats = model.get_image_features(**inputs)  # (N, d)

        feats = F.normalize(feats, p=2, dim=-1)
        feat_mean = feats.mean(dim=0)  # (d,)

        # Temperature scaling sharpens cosine similarities
        feat_scaled = feat_mean / temp
        feat_final = F.normalize(feat_scaled, p=2, dim=-1)

    return feat_final

############################## Choosing Images ##############################
def choose_image(
    target,
    sentence,
    images,
    image_dict,
    model: CLIPModel,
    processor: CLIPProcessor,
    photo_weight: float = 0.6,
    semantic_weight: float = 0.4,
    temp: float = 0.7,
    print_output: bool = False,
):
    """
    Choose the best matching image using:
    - dual-channel text embedding (photo + semantic prompts)
    - rich multi-view CLIP image embeddings
    """
    if not (isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel)):
        raise ValueError("Processor and model must be CLIPProcessor and CLIPModel.")

    device = next(model.parameters()).device

    # 1. Text embedding (dual-channel micro-prompt fusion)
    text_emb = get_dual_channel_text_embedding(
        target=target,
        sentence=sentence,
        processor=processor,
        model=model,
        photo_weight=photo_weight,
        semantic_weight=semantic_weight,
    )
    text_emb = text_emb.unsqueeze(0)  # (1, d)

    # 2. Image embeddings
    img_emb_list = []
    valid_names = []

    for name in images:
        valid_names.append(name)
        if name not in image_dict:
            d = model.config.projection_dim
            img_emb_list.append(torch.zeros(d, device=device))
            continue

        img = image_dict[name]
        emb = get_image_embedding(
            img,
            processor=processor,
            model=model,
            temp=temp,
        )
        img_emb_list.append(emb)

    img_feats = torch.stack(img_emb_list, dim=0)  # (N, d)

    # 3. Rank by cosine similarity
    sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
    ranked_indices = np.argsort(sims)[::-1]
    ranked_images = [valid_names[i] for i in ranked_indices]

    if print_output:
        for rank, i in enumerate(ranked_indices):
            name = valid_names[i]
            if name in image_dict:
                plt.imshow(image_dict[name])
                plt.title(f"Rank {rank+1} | sim={sims[i]:.4f}\nSentence: {sentence}")
                plt.axis("off")
                plt.show()
        print("Ranked Images:", ranked_images)

    ranked_captions = [None] * len(ranked_images)
    ranked_embs = [None] * len(ranked_images)
    return ranked_images, ranked_captions, ranked_embs

############################## Evaluation helpers ##############################
def evaluate_subset(
    df: pd.DataFrame,
    image_dict: dict,
    model: CLIPModel,
    processor: CLIPProcessor,
    photo_weight: float,
    semantic_weight: float,
    temp: float,
) -> tuple[float, float]:
    """
    Compute MRR and HitRate for a given subset of data.
    """
    ranks = []

    for _, row in df.iterrows():
        target = row["target"]
        sentence = row["sentence"]
        images = [row[f"image_{i}"] for i in range(10)]
        label = row["label"]

        ranked_images, _, _ = choose_image(
            target=target,
            sentence=sentence,
            images=images,
            image_dict=image_dict,
            model=model,
            processor=processor,
            photo_weight=photo_weight,
            semantic_weight=semantic_weight,
            temp=temp,
            print_output=False,
        )

        rank = ranked_images.index(label) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hit = np.mean(ranks == 1)
    return mrr, hit

############################## Optuna Objective ##############################
def objective(trial, val_data, image_dict, model, processor):
    """
    Optuna objective: sample hyperparameters and evaluate on fixed validation subset.
    Returns negative MRR so that direction='minimize' corresponds to maximizing MRR.
    """

    photo_weight = trial.suggest_float("photo_weight", 0.1, 0.9)
    semantic_weight = trial.suggest_float("semantic_weight", 0.1, 0.9)

    # enforce that they sum roughly to 1
    total = photo_weight + semantic_weight
    photo_weight /= total
    semantic_weight /= total

    temp = trial.suggest_float("temp", 0.4, 1.0)

    mrr, hit = evaluate_subset(
        val_data,
        image_dict=image_dict,
        model=model,
        processor=processor,
        photo_weight=photo_weight,
        semantic_weight=semantic_weight,
        temp=temp,
    )

    # You could also combine MRR + HitRate; here we just maximize MRR
    trial.set_user_attr("mrr", float(mrr))
    trial.set_user_attr("hit_rate", float(hit))

    # Optuna minimizes, so return negative MRR
    return -mrr

############################## Main ##############################
if __name__ == "__main__":

    # Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load Dataset
    file_path = "dataset"
    print_output = False

    # WordNet for synonyms
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # Load CLIP (currently ViT-B/32; change model_name if you want ViT-L/14)
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    print("Loading TRAIN and TEST datasets")
    train_data, train_image_dict = load_data(file_path=file_path, train_val="train")
    test_data, test_image_dict = load_data(file_path=file_path, train_val="test")


    # 80/20 Internal Split (for reference)
    np.random.seed(42)
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_df = train_data.iloc[train_idx].reset_index(drop=True)
    val_df = train_data.iloc[val_idx].reset_index(drop=True)

    print(f"[INFO] Training size: {len(train_df)} | Validation size: {len(val_df)}")

    # Bayesian Optimization (Optuna)
    print("\n==============================")
    print("Starting Bayesian tuning...")
    print("==============================")

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",  # because objective returns -MRR
        sampler=sampler,
    )

    N_TRIALS = 40  # adjust as you like (more = slower but better search)

    study.optimize(
        lambda trial: objective(
            trial,
            val_data=val_df,
            image_dict=train_image_dict,
            model=model,
            processor=processor,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # Print / Save Best Hyperparameters
    print("BEST BAYESIAN TRIAL")
    best_trial = study.best_trial
    best_params = best_trial.params
    print("Best params:", best_params)
    print("Best trial MRR:", -best_trial.value)
    print("Stored Hit Rate on best trial:", best_trial.user_attrs.get("hit_rate", None))

    with open("best_hyperparams.pkl", "wb") as f:
        pickle.dump(best_params, f)

    # Final Evaluation on Full Trial Set (using best hyperparameters)
    print("\nEvaluating full TRIAL set using best hyperparameters...")

    # Unpack best params
    photo_weight = best_params["photo_weight"]
    semantic_weight = best_params["semantic_weight"]
    total = photo_weight + semantic_weight
    photo_weight /= total
    semantic_weight /= total

    temp = best_params["temp"]

    final_mrr, final_hit = evaluate_subset(
        test_data,
        image_dict=test_image_dict,
        model=model,
        processor=processor,
        photo_weight=photo_weight,
        semantic_weight=semantic_weight,
        temp=temp,
    )

    # Print Evaluation Metrics
    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(f"MRR:      {final_mrr:.6f}")
    print(f"Hit Rate: {final_hit:.6f}")
