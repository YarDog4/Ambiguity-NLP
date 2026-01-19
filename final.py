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

nltk.data.path.insert(0, "/home/dbp52/Ambiguity-NLP/nltk_data")

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
    dtype = img_feats.dtype
    text_emb = text_emb.to(dtype)

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
    if not torch.cuda.is_available():
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    # Load CLIP (currently ViT-B/32; change model_name if you want ViT-L/14)
    if torch.cuda.is_available():   # If on the HPC, need to download this separately and add the file
        model_path = "/home/dbp52/Ambiguity-NLP/clip-vit-base-patch32"
        # model_path = "/home/dbp52/Ambiguity-NLP/clip-vit-base-patch16"
        # model_path = "/home/dbp52/Ambiguity-NLP/clip-vit-large-patch14"
        # model_path = "/home/dbp52/Ambiguity-NLP/CLIP-ViT-B-32-laion2B-s34B-b79K"
        print(f"Model_Path: {model_path}")
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
    else:
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
    print(f"Model_Path: {model_path}")







# ==============================
# Starting Bayesian tuning...     CLIP-VIT-Base_Patch32
# ==============================

#   0%|          | 0/40 [00:00<?, ?it/s]/home/dbp52/Ambiguity-NLP/final.py:708: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   ctx = torch.cuda.amp.autocast(enabled=use_amp)

                                      

#   0%|          | 0/40 [1:01:52<?, ?it/s]
# Best trial: 0. Best value: -0.849216:   0%|          | 0/40 [1:01:52<?, ?it/s]
# Best trial: 0. Best value: -0.849216:   2%|▎         | 1/40 [1:01:52<40:12:51, 3712.10s/it]
                                                                                           

# Best trial: 0. Best value: -0.849216:   2%|▎         | 1/40 [2:05:24<40:12:51, 3712.10s/it]
# Best trial: 1. Best value: -0.854894:   2%|▎         | 1/40 [2:05:24<40:12:51, 3712.10s/it]
# Best trial: 1. Best value: -0.854894:   5%|▌         | 2/40 [2:05:24<39:48:29, 3771.31s/it]
                                                                                           

# Best trial: 1. Best value: -0.854894:   5%|▌         | 2/40 [3:07:39<39:48:29, 3771.31s/it]
# Best trial: 1. Best value: -0.854894:   5%|▌         | 2/40 [3:07:39<39:48:29, 3771.31s/it]
# Best trial: 1. Best value: -0.854894:   8%|▊         | 3/40 [3:07:39<38:35:13, 3754.41s/it]
                                                                                           

# Best trial: 1. Best value: -0.854894:   8%|▊         | 3/40 [4:09:12<38:35:13, 3754.41s/it]
# Best trial: 1. Best value: -0.854894:   8%|▊         | 3/40 [4:09:12<38:35:13, 3754.41s/it]
# Best trial: 1. Best value: -0.854894:  10%|█         | 4/40 [4:09:12<37:18:07, 3730.20s/it]
                                                                                           

# Best trial: 1. Best value: -0.854894:  10%|█         | 4/40 [5:11:18<37:18:07, 3730.20s/it]
# Best trial: 4. Best value: -0.85669:  10%|█         | 4/40 [5:11:18<37:18:07, 3730.20s/it] 
# Best trial: 4. Best value: -0.85669:  12%|█▎        | 5/40 [5:11:18<36:15:10, 3728.88s/it]
                                                                                          

# Best trial: 4. Best value: -0.85669:  12%|█▎        | 5/40 [6:13:52<36:15:10, 3728.88s/it]
# Best trial: 4. Best value: -0.85669:  12%|█▎        | 5/40 [6:13:52<36:15:10, 3728.88s/it]
# Best trial: 4. Best value: -0.85669:  15%|█▌        | 6/40 [6:13:52<35:17:54, 3737.48s/it]
                                                                                          

# Best trial: 4. Best value: -0.85669:  15%|█▌        | 6/40 [7:16:50<35:17:54, 3737.48s/it]
# Best trial: 4. Best value: -0.85669:  15%|█▌        | 6/40 [7:16:50<35:17:54, 3737.48s/it]
# Best trial: 4. Best value: -0.85669:  18%|█▊        | 7/40 [7:16:50<34:22:52, 3750.70s/it]
                                                                                          

# Best trial: 4. Best value: -0.85669:  18%|█▊        | 7/40 [8:19:28<34:22:52, 3750.70s/it]
# Best trial: 4. Best value: -0.85669:  18%|█▊        | 7/40 [8:19:28<34:22:52, 3750.70s/it]
# Best trial: 4. Best value: -0.85669:  20%|██        | 8/40 [8:19:28<33:21:34, 3752.96s/it]
                                                                                          

# Best trial: 4. Best value: -0.85669:  20%|██        | 8/40 [9:21:41<33:21:34, 3752.96s/it]
# Best trial: 4. Best value: -0.85669:  20%|██        | 8/40 [9:21:41<33:21:34, 3752.96s/it]
# Best trial: 4. Best value: -0.85669:  22%|██▎       | 9/40 [9:21:41<32:15:44, 3746.58s/it]
                                                                                          

# Best trial: 4. Best value: -0.85669:  22%|██▎       | 9/40 [10:23:52<32:15:44, 3746.58s/it]
# Best trial: 4. Best value: -0.85669:  22%|██▎       | 9/40 [10:23:52<32:15:44, 3746.58s/it]
# Best trial: 4. Best value: -0.85669:  25%|██▌       | 10/40 [10:23:52<31:10:53, 3741.79s/it]
                                                                                            

# Best trial: 4. Best value: -0.85669:  25%|██▌       | 10/40 [11:25:23<31:10:53, 3741.79s/it]
# Best trial: 4. Best value: -0.85669:  25%|██▌       | 10/40 [11:25:23<31:10:53, 3741.79s/it]
# Best trial: 4. Best value: -0.85669:  28%|██▊       | 11/40 [11:25:23<30:00:59, 3726.17s/it]
                                                                                            

# Best trial: 4. Best value: -0.85669:  28%|██▊       | 11/40 [12:26:46<30:00:59, 3726.17s/it]
# Best trial: 4. Best value: -0.85669:  28%|██▊       | 11/40 [12:26:46<30:00:59, 3726.17s/it]
# Best trial: 4. Best value: -0.85669:  30%|███       | 12/40 [12:26:46<28:52:51, 3713.25s/it]
                                                                                            

# Best trial: 4. Best value: -0.85669:  30%|███       | 12/40 [13:29:34<28:52:51, 3713.25s/it]
# Best trial: 4. Best value: -0.85669:  30%|███       | 12/40 [13:29:34<28:52:51, 3713.25s/it]
# Best trial: 4. Best value: -0.85669:  32%|███▎      | 13/40 [13:29:34<27:58:22, 3729.73s/it]
                                                                                            

# Best trial: 4. Best value: -0.85669:  32%|███▎      | 13/40 [14:31:17<27:58:22, 3729.73s/it]
# Best trial: 4. Best value: -0.85669:  32%|███▎      | 13/40 [14:31:17<27:58:22, 3729.73s/it]
# Best trial: 4. Best value: -0.85669:  35%|███▌      | 14/40 [14:31:17<26:52:42, 3721.63s/it]
                                                                                            

# Best trial: 4. Best value: -0.85669:  35%|███▌      | 14/40 [15:33:20<26:52:42, 3721.63s/it]
# Best trial: 4. Best value: -0.85669:  35%|███▌      | 14/40 [15:33:20<26:52:42, 3721.63s/it]
# Best trial: 4. Best value: -0.85669:  38%|███▊      | 15/40 [15:33:20<25:50:49, 3721.98s/it]
                                                                                            

# Best trial: 4. Best value: -0.85669:  38%|███▊      | 15/40 [16:35:07<25:50:49, 3721.98s/it]
# Best trial: 4. Best value: -0.85669:  38%|███▊      | 15/40 [16:35:07<25:50:49, 3721.98s/it]
# Best trial: 4. Best value: -0.85669:  40%|████      | 16/40 [16:35:07<24:47:00, 3717.51s/it]
                                                                                            

# Best trial: 4. Best value: -0.85669:  40%|████      | 16/40 [17:36:40<24:47:00, 3717.51s/it]
# Best trial: 16. Best value: -0.857062:  40%|████      | 16/40 [17:36:40<24:47:00, 3717.51s/it]
# Best trial: 16. Best value: -0.857062:  42%|████▎     | 17/40 [17:36:40<23:42:17, 3710.33s/it]
                                                                                              

# Best trial: 16. Best value: -0.857062:  42%|████▎     | 17/40 [18:38:17<23:42:17, 3710.33s/it]
# Best trial: 17. Best value: -0.857094:  42%|████▎     | 17/40 [18:38:17<23:42:17, 3710.33s/it]
# Best trial: 17. Best value: -0.857094:  45%|████▌     | 18/40 [18:38:17<22:38:53, 3706.09s/it]
                                                                                              

# Best trial: 17. Best value: -0.857094:  45%|████▌     | 18/40 [19:39:47<22:38:53, 3706.09s/it]
# Best trial: 17. Best value: -0.857094:  45%|████▌     | 18/40 [19:39:47<22:38:53, 3706.09s/it]
# Best trial: 17. Best value: -0.857094:  48%|████▊     | 19/40 [19:39:47<21:35:30, 3701.44s/it]
                                                                                              

# Best trial: 17. Best value: -0.857094:  48%|████▊     | 19/40 [20:41:20<21:35:30, 3701.44s/it]
# Best trial: 19. Best value: -0.857479:  48%|████▊     | 19/40 [20:41:20<21:35:30, 3701.44s/it]
# Best trial: 19. Best value: -0.857479:  50%|█████     | 20/40 [20:41:20<20:32:58, 3698.95s/it]
                                                                                              

# Best trial: 19. Best value: -0.857479:  50%|█████     | 20/40 [21:42:58<20:32:58, 3698.95s/it]
# Best trial: 19. Best value: -0.857479:  50%|█████     | 20/40 [21:42:58<20:32:58, 3698.95s/it]
# Best trial: 19. Best value: -0.857479:  52%|█████▎    | 21/40 [21:42:58<19:31:14, 3698.66s/it]
                                                                                              

# Best trial: 19. Best value: -0.857479:  52%|█████▎    | 21/40 [22:44:32<19:31:14, 3698.66s/it]
# Best trial: 19. Best value: -0.857479:  52%|█████▎    | 21/40 [22:44:32<19:31:14, 3698.66s/it]
# Best trial: 19. Best value: -0.857479:  55%|█████▌    | 22/40 [22:44:32<18:29:06, 3697.04s/it]
                                                                                              

# Best trial: 19. Best value: -0.857479:  55%|█████▌    | 22/40 [23:46:03<18:29:06, 3697.04s/it]
# Best trial: 19. Best value: -0.857479:  55%|█████▌    | 22/40 [23:46:03<18:29:06, 3697.04s/it]
# Best trial: 19. Best value: -0.857479:  57%|█████▊    | 23/40 [23:46:03<17:27:02, 3695.44s/it]
                                                                                              

# Best trial: 19. Best value: -0.857479:  57%|█████▊    | 23/40 [24:47:42<17:27:02, 3695.44s/it]
# Best trial: 19. Best value: -0.857479:  57%|█████▊    | 23/40 [24:47:42<17:27:02, 3695.44s/it]
# Best trial: 19. Best value: -0.857479:  60%|██████    | 24/40 [24:47:42<16:25:40, 3696.28s/it]
                                                                                              

# Best trial: 19. Best value: -0.857479:  60%|██████    | 24/40 [25:49:19<16:25:40, 3696.28s/it]
# Best trial: 19. Best value: -0.857479:  60%|██████    | 24/40 [25:49:19<16:25:40, 3696.28s/it]
# Best trial: 19. Best value: -0.857479:  62%|██████▎   | 25/40 [25:49:19<15:24:07, 3696.53s/it]
                                                                                              

# Best trial: 19. Best value: -0.857479:  62%|██████▎   | 25/40 [26:50:58<15:24:07, 3696.53s/it]
# Best trial: 19. Best value: -0.857479:  62%|██████▎   | 25/40 [26:50:58<15:24:07, 3696.53s/it]
# Best trial: 19. Best value: -0.857479:  65%|██████▌   | 26/40 [26:50:58<14:22:42, 3697.30s/it]
                                                                                              

# Best trial: 19. Best value: -0.857479:  65%|██████▌   | 26/40 [27:52:32<14:22:42, 3697.30s/it]
# Best trial: 26. Best value: -0.857736:  65%|██████▌   | 26/40 [27:52:32<14:22:42, 3697.30s/it]
# Best trial: 26. Best value: -0.857736:  68%|██████▊   | 27/40 [27:52:32<13:20:53, 3696.46s/it]
                                                                                              

# Best trial: 26. Best value: -0.857736:  68%|██████▊   | 27/40 [28:54:04<13:20:53, 3696.46s/it]
# Best trial: 26. Best value: -0.857736:  68%|██████▊   | 27/40 [28:54:04<13:20:53, 3696.46s/it]
# Best trial: 26. Best value: -0.857736:  70%|███████   | 28/40 [28:54:04<12:19:00, 3695.03s/it]
                                                                                              

# Best trial: 26. Best value: -0.857736:  70%|███████   | 28/40 [29:55:28<12:19:00, 3695.03s/it]
# Best trial: 26. Best value: -0.857736:  70%|███████   | 28/40 [29:55:28<12:19:00, 3695.03s/it]
# Best trial: 26. Best value: -0.857736:  72%|███████▎  | 29/40 [29:55:28<11:16:48, 3691.65s/it]
                                                                                              

# Best trial: 26. Best value: -0.857736:  72%|███████▎  | 29/40 [30:56:59<11:16:48, 3691.65s/it]
# Best trial: 26. Best value: -0.857736:  72%|███████▎  | 29/40 [30:56:59<11:16:48, 3691.65s/it]
# Best trial: 26. Best value: -0.857736:  75%|███████▌  | 30/40 [30:56:59<10:15:14, 3691.44s/it]
                                                                                              

# Best trial: 26. Best value: -0.857736:  75%|███████▌  | 30/40 [31:58:25<10:15:14, 3691.44s/it]
# Best trial: 26. Best value: -0.857736:  75%|███████▌  | 30/40 [31:58:25<10:15:14, 3691.44s/it]
# Best trial: 26. Best value: -0.857736:  78%|███████▊  | 31/40 [31:58:25<9:13:30, 3690.03s/it] 
                                                                                             

# Best trial: 26. Best value: -0.857736:  78%|███████▊  | 31/40 [32:59:49<9:13:30, 3690.03s/it]
# Best trial: 31. Best value: -0.85792:  78%|███████▊  | 31/40 [32:59:49<9:13:30, 3690.03s/it] 
# Best trial: 31. Best value: -0.85792:  80%|████████  | 32/40 [32:59:49<8:11:44, 3688.02s/it]
                                                                                            

# Best trial: 31. Best value: -0.85792:  80%|████████  | 32/40 [34:01:26<8:11:44, 3688.02s/it]
# Best trial: 31. Best value: -0.85792:  80%|████████  | 32/40 [34:01:26<8:11:44, 3688.02s/it]
# Best trial: 31. Best value: -0.85792:  82%|████████▎ | 33/40 [34:01:26<7:10:35, 3690.82s/it]
                                                                                            

# Best trial: 31. Best value: -0.85792:  82%|████████▎ | 33/40 [35:02:58<7:10:35, 3690.82s/it]
# Best trial: 31. Best value: -0.85792:  82%|████████▎ | 33/40 [35:02:58<7:10:35, 3690.82s/it]
# Best trial: 31. Best value: -0.85792:  85%|████████▌ | 34/40 [35:02:58<6:09:07, 3691.27s/it]
                                                                                            

# Best trial: 31. Best value: -0.85792:  85%|████████▌ | 34/40 [36:04:30<6:09:07, 3691.27s/it]
# Best trial: 31. Best value: -0.85792:  85%|████████▌ | 34/40 [36:04:30<6:09:07, 3691.27s/it]
# Best trial: 31. Best value: -0.85792:  88%|████████▊ | 35/40 [36:04:30<5:07:37, 3691.48s/it]
                                                                                            

# Best trial: 31. Best value: -0.85792:  88%|████████▊ | 35/40 [37:05:56<5:07:37, 3691.48s/it]
# Best trial: 31. Best value: -0.85792:  88%|████████▊ | 35/40 [37:05:56<5:07:37, 3691.48s/it]
# Best trial: 31. Best value: -0.85792:  90%|█████████ | 36/40 [37:05:56<4:05:59, 3689.86s/it]
                                                                                            

# Best trial: 31. Best value: -0.85792:  90%|█████████ | 36/40 [38:07:32<4:05:59, 3689.86s/it]
# Best trial: 31. Best value: -0.85792:  90%|█████████ | 36/40 [38:07:32<4:05:59, 3689.86s/it]
# Best trial: 31. Best value: -0.85792:  92%|█████████▎| 37/40 [38:07:32<3:04:34, 3691.46s/it]
                                                                                            

# Best trial: 31. Best value: -0.85792:  92%|█████████▎| 37/40 [39:09:05<3:04:34, 3691.46s/it]
# Best trial: 31. Best value: -0.85792:  92%|█████████▎| 37/40 [39:09:05<3:04:34, 3691.46s/it]
# Best trial: 31. Best value: -0.85792:  95%|█████████▌| 38/40 [39:09:05<2:03:03, 3691.93s/it]
                                                                                            

# Best trial: 31. Best value: -0.85792:  95%|█████████▌| 38/40 [40:10:36<2:03:03, 3691.93s/it]
# Best trial: 31. Best value: -0.85792:  95%|█████████▌| 38/40 [40:10:36<2:03:03, 3691.93s/it]
# Best trial: 31. Best value: -0.85792:  98%|█████████▊| 39/40 [40:10:36<1:01:31, 3691.70s/it]
                                                                                            

# Best trial: 31. Best value: -0.85792:  98%|█████████▊| 39/40 [41:12:04<1:01:31, 3691.70s/it]
# Best trial: 31. Best value: -0.85792:  98%|█████████▊| 39/40 [41:12:04<1:01:31, 3691.70s/it]
# Best trial: 31. Best value: -0.85792: 100%|██████████| 40/40 [41:12:04<00:00, 3690.55s/it]  
# Best trial: 31. Best value: -0.85792: 100%|██████████| 40/40 [41:12:04<00:00, 3708.10s/it]
# [I 2026-01-15 14:34:40,475] Trial 0 finished with value: -0.8492160617160617 and parameters: {'photo_weight': 0.39963209507789, 'semantic_weight': 0.8605714451279329, 'temp': 0.839196365086843}. Best is trial 0 with value: -0.8492160617160617.
# [I 2026-01-15 15:38:13,232] Trial 1 finished with value: -0.8548938715605382 and parameters: {'photo_weight': 0.5789267873576293, 'semantic_weight': 0.22481491235394924, 'temp': 0.49359671220172163}. Best is trial 1 with value: -0.8548938715605382.
# [I 2026-01-15 16:40:27,527] Trial 2 finished with value: -0.8396885521885523 and parameters: {'photo_weight': 0.14646688973455957, 'semantic_weight': 0.7929409166199481, 'temp': 0.7606690070459252}. Best is trial 1 with value: -0.8548938715605382.
# [I 2026-01-15 17:42:00,620] Trial 3 finished with value: -0.8539906081572749 and parameters: {'photo_weight': 0.6664580622368363, 'semantic_weight': 0.11646759543664197, 'temp': 0.9819459112971965}. Best is trial 1 with value: -0.8548938715605382.
# [I 2026-01-15 18:44:07,153] Trial 4 finished with value: -0.85669022335689 and parameters: {'photo_weight': 0.7659541126403374, 'semantic_weight': 0.26987128854262094, 'temp': 0.5090949803242604}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-15 19:46:41,330] Trial 5 finished with value: -0.8532112948779617 and parameters: {'photo_weight': 0.24672360788274705, 'semantic_weight': 0.3433937943676302, 'temp': 0.7148538589793427}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-15 20:49:39,232] Trial 6 finished with value: -0.8563305521638854 and parameters: {'photo_weight': 0.4455560149136927, 'semantic_weight': 0.3329833121584336, 'temp': 0.7671117368334277}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-15 21:52:17,045] Trial 7 finished with value: -0.8516441891441892 and parameters: {'photo_weight': 0.21159508852163347, 'semantic_weight': 0.3337157188281745, 'temp': 0.619817105976215}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-15 22:54:29,599] Trial 8 finished with value: -0.851951751951752 and parameters: {'photo_weight': 0.46485598737362877, 'semantic_weight': 0.728140769114411, 'temp': 0.5198042692950159}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-15 23:56:40,655] Trial 9 finished with value: -0.8538100788100788 and parameters: {'photo_weight': 0.5113875507308893, 'semantic_weight': 0.573931655089634, 'temp': 0.42787024763199866}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-16 00:58:11,420] Trial 10 finished with value: -0.8563270063270063 and parameters: {'photo_weight': 0.8823968592649603, 'semantic_weight': 0.5470807186059738, 'temp': 0.5863511074356347}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-16 01:59:35,121] Trial 11 finished with value: -0.855396917896918 and parameters: {'photo_weight': 0.7963949118197544, 'semantic_weight': 0.3768282196663541, 'temp': 0.8399190066427125}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-16 03:02:22,767] Trial 12 finished with value: -0.8553261553261554 and parameters: {'photo_weight': 0.32830741213119097, 'semantic_weight': 0.17189049400247203, 'temp': 0.6054276085964654}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-16 04:04:05,681] Trial 13 finished with value: -0.8551177217843884 and parameters: {'photo_weight': 0.6907244357800169, 'semantic_weight': 0.4505236824571834, 'temp': 0.8182935390040567}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-16 05:06:08,457] Trial 14 finished with value: -0.8562468395801729 and parameters: {'photo_weight': 0.6554437608969272, 'semantic_weight': 0.24486913201015623, 'temp': 0.9389534489284183}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-16 06:07:55,590] Trial 15 finished with value: -0.8535520035520037 and parameters: {'photo_weight': 0.35462900009984644, 'semantic_weight': 0.4416898445066654, 'temp': 0.4143445254690108}. Best is trial 4 with value: -0.85669022335689.
# [I 2026-01-16 07:09:29,225] Trial 16 finished with value: -0.8570617653950986 and parameters: {'photo_weight': 0.7673851060844344, 'semantic_weight': 0.6176251544533542, 'temp': 0.6710635518971032}. Best is trial 16 with value: -0.8570617653950986.
# [I 2026-01-16 08:11:05,442] Trial 17 finished with value: -0.8570941404274738 and parameters: {'photo_weight': 0.7904482457053954, 'semantic_weight': 0.6486418285419191, 'temp': 0.6689923975930208}. Best is trial 17 with value: -0.8570941404274738.
# [I 2026-01-16 09:12:36,055] Trial 18 finished with value: -0.8561877936877936 and parameters: {'photo_weight': 0.8892255827201372, 'semantic_weight': 0.6572655446836505, 'temp': 0.6728815060081761}. Best is trial 17 with value: -0.8570941404274738.
# [I 2026-01-16 10:14:09,197] Trial 19 finished with value: -0.8574787866454535 and parameters: {'photo_weight': 0.769724806039641, 'semantic_weight': 0.6208126425372757, 'temp': 0.6739260139905778}. Best is trial 19 with value: -0.8574787866454535.
# [I 2026-01-16 11:15:47,199] Trial 20 finished with value: -0.8533688533688533 and parameters: {'photo_weight': 0.5691479398363706, 'semantic_weight': 0.7028618708432064, 'temp': 0.574888577100112}. Best is trial 19 with value: -0.8574787866454535.
# [I 2026-01-16 12:17:20,464] Trial 21 finished with value: -0.8567406359073025 and parameters: {'photo_weight': 0.7830211978188072, 'semantic_weight': 0.6160080521242813, 'temp': 0.6732259667579987}. Best is trial 19 with value: -0.8574787866454535.
# [I 2026-01-16 13:18:52,170] Trial 22 finished with value: -0.856863352696686 and parameters: {'photo_weight': 0.7254381587860342, 'semantic_weight': 0.5123160815694576, 'temp': 0.7180141040500321}. Best is trial 19 with value: -0.8574787866454535.
# [I 2026-01-16 14:20:30,390] Trial 23 finished with value: -0.8558654308654309 and parameters: {'photo_weight': 0.8390792557268122, 'semantic_weight': 0.7782610685483555, 'temp': 0.645875324632317}. Best is trial 19 with value: -0.8574787866454535.
# [I 2026-01-16 15:22:07,508] Trial 24 finished with value: -0.8544253585920251 and parameters: {'photo_weight': 0.6062939755042287, 'semantic_weight': 0.6399329547973525, 'temp': 0.7783547384931431}. Best is trial 19 with value: -0.8574787866454535.
# [I 2026-01-16 16:23:46,597] Trial 25 finished with value: -0.8545326586993254 and parameters: {'photo_weight': 0.8284567510131701, 'semantic_weight': 0.8920160949902591, 'temp': 0.7084270039293292}. Best is trial 19 with value: -0.8574787866454535.
# [I 2026-01-16 17:25:21,107] Trial 26 finished with value: -0.8577362452362453 and parameters: {'photo_weight': 0.7260476515295164, 'semantic_weight': 0.5869043676185652, 'temp': 0.5601422512453316}. Best is trial 26 with value: -0.8577362452362453.
# [I 2026-01-16 18:26:52,804] Trial 27 finished with value: -0.8561746895080228 and parameters: {'photo_weight': 0.703288847643559, 'semantic_weight': 0.4558248580336193, 'temp': 0.5599656191401006}. Best is trial 26 with value: -0.8577362452362453.
# [I 2026-01-16 19:28:16,552] Trial 28 finished with value: -0.8537447120780456 and parameters: {'photo_weight': 0.6099585878008916, 'semantic_weight': 0.6976570846125196, 'temp': 0.545895636749907}. Best is trial 26 with value: -0.8577362452362453.
# [I 2026-01-16 20:29:47,494] Trial 29 finished with value: -0.8531130906130906 and parameters: {'photo_weight': 0.7392309506772959, 'semantic_weight': 0.8307000624641604, 'temp': 0.47262254404644166}. Best is trial 26 with value: -0.8577362452362453.
# [I 2026-01-16 21:31:14,257] Trial 30 finished with value: -0.8550625300625301 and parameters: {'photo_weight': 0.8341516582694036, 'semantic_weight': 0.5345109376159397, 'temp': 0.8654949316297752}. Best is trial 26 with value: -0.8577362452362453.
# [I 2026-01-16 22:32:37,593] Trial 31 finished with value: -0.8579197037530373 and parameters: {'photo_weight': 0.7684612912924019, 'semantic_weight': 0.615171505253044, 'temp': 0.6335228783780633}. Best is trial 31 with value: -0.8579197037530373.
# [I 2026-01-16 23:34:14,948] Trial 32 finished with value: -0.8557353140686473 and parameters: {'photo_weight': 0.6535941482502321, 'semantic_weight': 0.5817532175577731, 'temp': 0.631182336578}. Best is trial 31 with value: -0.8579197037530373.
# [I 2026-01-17 00:35:47,266] Trial 33 finished with value: -0.856522644022644 and parameters: {'photo_weight': 0.825157757305528, 'semantic_weight': 0.7429650292238459, 'temp': 0.7436132684639093}. Best is trial 31 with value: -0.8579197037530373.
# [I 2026-01-17 01:37:19,218] Trial 34 finished with value: -0.8529900654900655 and parameters: {'photo_weight': 0.5296108011704279, 'semantic_weight': 0.6642796194958729, 'temp': 0.4662004417882814}. Best is trial 31 with value: -0.8579197037530373.
# [I 2026-01-17 02:38:45,313] Trial 35 finished with value: -0.8558119349786015 and parameters: {'photo_weight': 0.7320079634978758, 'semantic_weight': 0.4894048086194761, 'temp': 0.5335619892045866}. Best is trial 31 with value: -0.8579197037530373.
# [I 2026-01-17 03:40:20,501] Trial 36 finished with value: -0.8569452152785486 and parameters: {'photo_weight': 0.8915153388294712, 'semantic_weight': 0.597914287920288, 'temp': 0.6490748878388283}. Best is trial 31 with value: -0.8579197037530373.
# [I 2026-01-17 04:41:53,532] Trial 37 finished with value: -0.8529160654160655 and parameters: {'photo_weight': 0.6323435954015886, 'semantic_weight': 0.8186386134551333, 'temp': 0.6041732284795238}. Best is trial 31 with value: -0.8579197037530373.
# [I 2026-01-17 05:43:24,694] Trial 38 finished with value: -0.8531084656084656 and parameters: {'photo_weight': 0.6884403642224474, 'semantic_weight': 0.7645510271239836, 'temp': 0.7382440387740647}. Best is trial 31 with value: -0.8579197037530373.
# [I 2026-01-17 06:44:52,553] Trial 39 finished with value: -0.8565642690642692 and parameters: {'photo_weight': 0.7945921618524117, 'semantic_weight': 0.7020396746621943, 'temp': 0.678310446732422}. Best is trial 31 with value: -0.8579197037530373.
# BEST BAYESIAN TRIAL
# Best params: {'photo_weight': 0.7684612912924019, 'semantic_weight': 0.615171505253044, 'temp': 0.6335228783780633}
# Best trial MRR: 0.8579197037530373
# Stored Hit Rate on best trial: 0.7793317793317793

# Evaluating full TRIAL set using best hyperparameters...

# ==============================
# FINAL RESULTS
# ==============================
# MRR:      0.739190
# Hit Rate: 0.593952












# ==============================
# Starting Bayesian tuning...    CLIP-VIT-Base_Patch32
# ==============================

#   0%|          | 0/40 [00:00<?, ?it/s]/home/dbp52/Ambiguity-NLP/final.py:710: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   ctx = torch.cuda.amp.autocast(enabled=use_amp)

                                      

#   0%|          | 0/40 [1:17:20<?, ?it/s]
# Best trial: 0. Best value: -0.850548:   0%|          | 0/40 [1:17:20<?, ?it/s]
# Best trial: 0. Best value: -0.850548:   2%|▎         | 1/40 [1:17:20<50:16:07, 4640.19s/it]
                                                                                           

# Best trial: 0. Best value: -0.850548:   2%|▎         | 1/40 [2:34:36<50:16:07, 4640.19s/it]
# Best trial: 1. Best value: -0.858709:   2%|▎         | 1/40 [2:34:36<50:16:07, 4640.19s/it]
# Best trial: 1. Best value: -0.858709:   5%|▌         | 2/40 [2:34:36<48:57:30, 4638.16s/it]
                                                                                           

# Best trial: 1. Best value: -0.858709:   5%|▌         | 2/40 [3:51:58<48:57:30, 4638.16s/it]
# Best trial: 1. Best value: -0.858709:   5%|▌         | 2/40 [3:51:58<48:57:30, 4638.16s/it]
# Best trial: 1. Best value: -0.858709:   8%|▊         | 3/40 [3:51:58<47:41:10, 4639.75s/it]
                                                                                           

# Best trial: 1. Best value: -0.858709:   8%|▊         | 3/40 [5:09:13<47:41:10, 4639.75s/it]
# Best trial: 1. Best value: -0.858709:   8%|▊         | 3/40 [5:09:13<47:41:10, 4639.75s/it]
# Best trial: 1. Best value: -0.858709:  10%|█         | 4/40 [5:09:13<46:22:48, 4638.02s/it]
                                                                                           

# Best trial: 1. Best value: -0.858709:  10%|█         | 4/40 [6:26:29<46:22:48, 4638.02s/it]
# Best trial: 1. Best value: -0.858709:  10%|█         | 4/40 [6:26:29<46:22:48, 4638.02s/it]
# Best trial: 1. Best value: -0.858709:  12%|█▎        | 5/40 [6:26:29<45:04:53, 4636.96s/it]
                                                                                           

# Best trial: 1. Best value: -0.858709:  12%|█▎        | 5/40 [7:43:44<45:04:53, 4636.96s/it]
# Best trial: 1. Best value: -0.858709:  12%|█▎        | 5/40 [7:43:44<45:04:53, 4636.96s/it]
# Best trial: 1. Best value: -0.858709:  15%|█▌        | 6/40 [7:43:44<43:47:22, 4636.55s/it]
                                                                                           

# Best trial: 1. Best value: -0.858709:  15%|█▌        | 6/40 [9:01:02<43:47:22, 4636.55s/it]
# Best trial: 1. Best value: -0.858709:  15%|█▌        | 6/40 [9:01:02<43:47:22, 4636.55s/it]
# Best trial: 1. Best value: -0.858709:  18%|█▊        | 7/40 [9:01:02<42:30:14, 4636.82s/it]
                                                                                           

# Best trial: 1. Best value: -0.858709:  18%|█▊        | 7/40 [10:18:19<42:30:14, 4636.82s/it]
# Best trial: 1. Best value: -0.858709:  18%|█▊        | 7/40 [10:18:19<42:30:14, 4636.82s/it]
# Best trial: 1. Best value: -0.858709:  20%|██        | 8/40 [10:18:19<41:13:06, 4637.08s/it]
                                                                                            

# Best trial: 1. Best value: -0.858709:  20%|██        | 8/40 [11:35:39<41:13:06, 4637.08s/it]
# Best trial: 1. Best value: -0.858709:  20%|██        | 8/40 [11:35:39<41:13:06, 4637.08s/it]
# Best trial: 1. Best value: -0.858709:  22%|██▎       | 9/40 [11:35:39<39:56:13, 4637.86s/it]
                                                                                            

# Best trial: 1. Best value: -0.858709:  22%|██▎       | 9/40 [12:52:57<39:56:13, 4637.86s/it]
# Best trial: 1. Best value: -0.858709:  22%|██▎       | 9/40 [12:52:57<39:56:13, 4637.86s/it]
# Best trial: 1. Best value: -0.858709:  25%|██▌       | 10/40 [12:52:57<38:38:58, 4637.97s/it]
                                                                                             

# Best trial: 1. Best value: -0.858709:  25%|██▌       | 10/40 [14:10:16<38:38:58, 4637.97s/it]
# Best trial: 1. Best value: -0.858709:  25%|██▌       | 10/40 [14:10:16<38:38:58, 4637.97s/it]
# Best trial: 1. Best value: -0.858709:  28%|██▊       | 11/40 [14:10:16<37:21:50, 4638.30s/it]
                                                                                             

# Best trial: 1. Best value: -0.858709:  28%|██▊       | 11/40 [15:27:37<37:21:50, 4638.30s/it]
# Best trial: 11. Best value: -0.858937:  28%|██▊       | 11/40 [15:27:37<37:21:50, 4638.30s/it]
# Best trial: 11. Best value: -0.858937:  30%|███       | 12/40 [15:27:37<36:04:52, 4639.02s/it]
                                                                                              

# Best trial: 11. Best value: -0.858937:  30%|███       | 12/40 [16:44:56<36:04:52, 4639.02s/it]
# Best trial: 12. Best value: -0.858982:  30%|███       | 12/40 [16:44:56<36:04:52, 4639.02s/it]
# Best trial: 12. Best value: -0.858982:  32%|███▎      | 13/40 [16:44:56<34:47:34, 4639.07s/it]
                                                                                              

# Best trial: 12. Best value: -0.858982:  32%|███▎      | 13/40 [18:02:15<34:47:34, 4639.07s/it]
# Best trial: 12. Best value: -0.858982:  32%|███▎      | 13/40 [18:02:15<34:47:34, 4639.07s/it]
# Best trial: 12. Best value: -0.858982:  35%|███▌      | 14/40 [18:02:15<33:30:16, 4639.08s/it]
                                                                                              

# Best trial: 12. Best value: -0.858982:  35%|███▌      | 14/40 [19:19:30<33:30:16, 4639.08s/it]
# Best trial: 14. Best value: -0.859036:  35%|███▌      | 14/40 [19:19:30<33:30:16, 4639.08s/it]
# Best trial: 14. Best value: -0.859036:  38%|███▊      | 15/40 [19:19:30<32:12:26, 4637.87s/it]
                                                                                              

# Best trial: 14. Best value: -0.859036:  38%|███▊      | 15/40 [20:36:44<32:12:26, 4637.87s/it]
# Best trial: 14. Best value: -0.859036:  38%|███▊      | 15/40 [20:36:44<32:12:26, 4637.87s/it]
# Best trial: 14. Best value: -0.859036:  40%|████      | 16/40 [20:36:44<30:54:40, 4636.68s/it]
                                                                                              

# Best trial: 14. Best value: -0.859036:  40%|████      | 16/40 [21:53:59<30:54:40, 4636.68s/it]
# Best trial: 16. Best value: -0.859099:  40%|████      | 16/40 [21:53:59<30:54:40, 4636.68s/it]
# Best trial: 16. Best value: -0.859099:  42%|████▎     | 17/40 [21:53:59<29:37:08, 4636.02s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  42%|████▎     | 17/40 [23:11:15<29:37:08, 4636.02s/it]
# Best trial: 16. Best value: -0.859099:  42%|████▎     | 17/40 [23:11:15<29:37:08, 4636.02s/it]
# Best trial: 16. Best value: -0.859099:  45%|████▌     | 18/40 [23:11:15<28:19:52, 4636.03s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  45%|████▌     | 18/40 [24:28:29<28:19:52, 4636.03s/it]
# Best trial: 16. Best value: -0.859099:  45%|████▌     | 18/40 [24:28:29<28:19:52, 4636.03s/it]
# Best trial: 16. Best value: -0.859099:  48%|████▊     | 19/40 [24:28:29<27:02:23, 4635.40s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  48%|████▊     | 19/40 [25:45:42<27:02:23, 4635.40s/it]
# Best trial: 16. Best value: -0.859099:  48%|████▊     | 19/40 [25:45:42<27:02:23, 4635.40s/it]
# Best trial: 16. Best value: -0.859099:  50%|█████     | 20/40 [25:45:42<25:44:58, 4634.94s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  50%|█████     | 20/40 [27:02:57<25:44:58, 4634.94s/it]
# Best trial: 16. Best value: -0.859099:  50%|█████     | 20/40 [27:02:57<25:44:58, 4634.94s/it]
# Best trial: 16. Best value: -0.859099:  52%|█████▎    | 21/40 [27:02:57<24:27:43, 4634.94s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  52%|█████▎    | 21/40 [28:20:12<24:27:43, 4634.94s/it]
# Best trial: 16. Best value: -0.859099:  52%|█████▎    | 21/40 [28:20:12<24:27:43, 4634.94s/it]
# Best trial: 16. Best value: -0.859099:  55%|█████▌    | 22/40 [28:20:12<23:10:25, 4634.75s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  55%|█████▌    | 22/40 [29:37:26<23:10:25, 4634.75s/it]
# Best trial: 16. Best value: -0.859099:  55%|█████▌    | 22/40 [29:37:26<23:10:25, 4634.75s/it]
# Best trial: 16. Best value: -0.859099:  57%|█████▊    | 23/40 [29:37:26<21:53:10, 4634.75s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  57%|█████▊    | 23/40 [30:54:41<21:53:10, 4634.75s/it]
# Best trial: 16. Best value: -0.859099:  57%|█████▊    | 23/40 [30:54:41<21:53:10, 4634.75s/it]
# Best trial: 16. Best value: -0.859099:  60%|██████    | 24/40 [30:54:41<20:35:55, 4634.72s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  60%|██████    | 24/40 [32:11:55<20:35:55, 4634.72s/it]
# Best trial: 16. Best value: -0.859099:  60%|██████    | 24/40 [32:11:55<20:35:55, 4634.72s/it]
# Best trial: 16. Best value: -0.859099:  62%|██████▎   | 25/40 [32:11:55<19:18:38, 4634.59s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  62%|██████▎   | 25/40 [33:29:10<19:18:38, 4634.59s/it]
# Best trial: 16. Best value: -0.859099:  62%|██████▎   | 25/40 [33:29:10<19:18:38, 4634.59s/it]
# Best trial: 16. Best value: -0.859099:  65%|██████▌   | 26/40 [33:29:10<18:01:24, 4634.60s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  65%|██████▌   | 26/40 [34:46:25<18:01:24, 4634.60s/it]
# Best trial: 16. Best value: -0.859099:  65%|██████▌   | 26/40 [34:46:25<18:01:24, 4634.60s/it]
# Best trial: 16. Best value: -0.859099:  68%|██████▊   | 27/40 [34:46:25<16:44:09, 4634.60s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  68%|██████▊   | 27/40 [36:03:38<16:44:09, 4634.60s/it]
# Best trial: 16. Best value: -0.859099:  68%|██████▊   | 27/40 [36:03:38<16:44:09, 4634.60s/it]
# Best trial: 16. Best value: -0.859099:  70%|███████   | 28/40 [36:03:38<15:26:51, 4634.31s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  70%|███████   | 28/40 [37:20:53<15:26:51, 4634.31s/it]
# Best trial: 16. Best value: -0.859099:  70%|███████   | 28/40 [37:20:53<15:26:51, 4634.31s/it]
# Best trial: 16. Best value: -0.859099:  72%|███████▎  | 29/40 [37:20:53<14:09:39, 4634.50s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  72%|███████▎  | 29/40 [38:38:09<14:09:39, 4634.50s/it]
# Best trial: 16. Best value: -0.859099:  72%|███████▎  | 29/40 [38:38:09<14:09:39, 4634.50s/it]
# Best trial: 16. Best value: -0.859099:  75%|███████▌  | 30/40 [38:38:09<12:52:29, 4634.97s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  75%|███████▌  | 30/40 [39:55:25<12:52:29, 4634.97s/it]
# Best trial: 16. Best value: -0.859099:  75%|███████▌  | 30/40 [39:55:25<12:52:29, 4634.97s/it]
# Best trial: 16. Best value: -0.859099:  78%|███████▊  | 31/40 [39:55:25<11:35:15, 4635.09s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  78%|███████▊  | 31/40 [41:12:42<11:35:15, 4635.09s/it]
# Best trial: 16. Best value: -0.859099:  78%|███████▊  | 31/40 [41:12:42<11:35:15, 4635.09s/it]
# Best trial: 16. Best value: -0.859099:  80%|████████  | 32/40 [41:12:42<10:18:07, 4635.90s/it]
                                                                                              

# Best trial: 16. Best value: -0.859099:  80%|████████  | 32/40 [42:29:58<10:18:07, 4635.90s/it]
# Best trial: 16. Best value: -0.859099:  80%|████████  | 32/40 [42:29:58<10:18:07, 4635.90s/it]
# Best trial: 16. Best value: -0.859099:  82%|████████▎ | 33/40 [42:29:58<9:00:50, 4635.85s/it] 
                                                                                             

# Best trial: 16. Best value: -0.859099:  82%|████████▎ | 33/40 [43:47:14<9:00:50, 4635.85s/it]
# Best trial: 33. Best value: -0.859135:  82%|████████▎ | 33/40 [43:47:14<9:00:50, 4635.85s/it]
# Best trial: 33. Best value: -0.859135:  85%|████████▌ | 34/40 [43:47:14<7:43:35, 4635.95s/it]
                                                                                             

# Best trial: 33. Best value: -0.859135:  85%|████████▌ | 34/40 [45:04:30<7:43:35, 4635.95s/it]
# Best trial: 33. Best value: -0.859135:  85%|████████▌ | 34/40 [45:04:30<7:43:35, 4635.95s/it]
# Best trial: 33. Best value: -0.859135:  88%|████████▊ | 35/40 [45:04:30<6:26:19, 4635.81s/it]
                                                                                             

# Best trial: 33. Best value: -0.859135:  88%|████████▊ | 35/40 [46:21:45<6:26:19, 4635.81s/it]
# Best trial: 35. Best value: -0.859441:  88%|████████▊ | 35/40 [46:21:45<6:26:19, 4635.81s/it]
# Best trial: 35. Best value: -0.859441:  90%|█████████ | 36/40 [46:21:45<5:09:02, 4635.54s/it]
                                                                                             

# Best trial: 35. Best value: -0.859441:  90%|█████████ | 36/40 [47:39:03<5:09:02, 4635.54s/it]
# Best trial: 35. Best value: -0.859441:  90%|█████████ | 36/40 [47:39:03<5:09:02, 4635.54s/it]
# Best trial: 35. Best value: -0.859441:  92%|█████████▎| 37/40 [47:39:03<3:51:49, 4636.46s/it]
                                                                                             

# Best trial: 35. Best value: -0.859441:  92%|█████████▎| 37/40 [48:56:20<3:51:49, 4636.46s/it]
# Best trial: 35. Best value: -0.859441:  92%|█████████▎| 37/40 [48:56:20<3:51:49, 4636.46s/it]
# Best trial: 35. Best value: -0.859441:  95%|█████████▌| 38/40 [48:56:20<2:34:32, 4636.50s/it]
                                                                                             

# Best trial: 35. Best value: -0.859441:  95%|█████████▌| 38/40 [50:13:37<2:34:32, 4636.50s/it]
# Best trial: 38. Best value: -0.859727:  95%|█████████▌| 38/40 [50:13:37<2:34:32, 4636.50s/it]
# Best trial: 38. Best value: -0.859727:  98%|█████████▊| 39/40 [50:13:37<1:17:16, 4636.69s/it]
                                                                                             

# Best trial: 38. Best value: -0.859727:  98%|█████████▊| 39/40 [51:30:54<1:17:16, 4636.69s/it]
# Best trial: 38. Best value: -0.859727:  98%|█████████▊| 39/40 [51:30:54<1:17:16, 4636.69s/it]
# Best trial: 38. Best value: -0.859727: 100%|██████████| 40/40 [51:30:54<00:00, 4636.68s/it]  
# Best trial: 38. Best value: -0.859727: 100%|██████████| 40/40 [51:30:54<00:00, 4636.35s/it]
# [I 2026-01-15 17:52:08,307] Trial 0 finished with value: -0.8505480630480631 and parameters: {'photo_weight': 0.39963209507789, 'semantic_weight': 0.8605714451279329, 'temp': 0.839196365086843}. Best is trial 0 with value: -0.8505480630480631.
# [I 2026-01-15 19:09:25,050] Trial 1 finished with value: -0.8587091920425254 and parameters: {'photo_weight': 0.5789267873576293, 'semantic_weight': 0.22481491235394924, 'temp': 0.49359671220172163}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-15 20:26:46,693] Trial 2 finished with value: -0.8485064318397653 and parameters: {'photo_weight': 0.14646688973455957, 'semantic_weight': 0.7929409166199481, 'temp': 0.7606690070459252}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-15 21:44:02,054] Trial 3 finished with value: -0.858649529482863 and parameters: {'photo_weight': 0.6664580622368363, 'semantic_weight': 0.11646759543664197, 'temp': 0.9819459112971965}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-15 23:01:17,125] Trial 4 finished with value: -0.8583823583823585 and parameters: {'photo_weight': 0.7659541126403374, 'semantic_weight': 0.26987128854262094, 'temp': 0.5090949803242604}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-16 00:18:32,893] Trial 5 finished with value: -0.8548248048248049 and parameters: {'photo_weight': 0.24672360788274705, 'semantic_weight': 0.3433937943676302, 'temp': 0.7148538589793427}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-16 01:35:50,251] Trial 6 finished with value: -0.8568618110284777 and parameters: {'photo_weight': 0.4455560149136927, 'semantic_weight': 0.3329833121584336, 'temp': 0.7671117368334277}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-16 02:53:07,912] Trial 7 finished with value: -0.8543660043660043 and parameters: {'photo_weight': 0.21159508852163347, 'semantic_weight': 0.3337157188281745, 'temp': 0.619817105976215}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-16 04:10:27,491] Trial 8 finished with value: -0.854863654863655 and parameters: {'photo_weight': 0.46485598737362877, 'semantic_weight': 0.728140769114411, 'temp': 0.5198042692950159}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-16 05:27:45,686] Trial 9 finished with value: -0.8553611511944844 and parameters: {'photo_weight': 0.5113875507308893, 'semantic_weight': 0.573931655089634, 'temp': 0.42787024763199866}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-16 06:45:04,754] Trial 10 finished with value: -0.8586371961371961 and parameters: {'photo_weight': 0.8628722031102568, 'semantic_weight': 0.10479774636465367, 'temp': 0.40452839158977594}. Best is trial 1 with value: -0.8587091920425254.
# [I 2026-01-16 08:02:25,418] Trial 11 finished with value: -0.8589370506037173 and parameters: {'photo_weight': 0.659270955808696, 'semantic_weight': 0.11229418986248343, 'temp': 0.9445988771695106}. Best is trial 11 with value: -0.8589370506037173.
# [I 2026-01-16 09:19:44,586] Trial 12 finished with value: -0.8589819131485797 and parameters: {'photo_weight': 0.6553141677274327, 'semantic_weight': 0.2047113709865967, 'temp': 0.9681771258340897}. Best is trial 12 with value: -0.8589819131485797.
# [I 2026-01-16 10:37:03,700] Trial 13 finished with value: -0.8575119325119326 and parameters: {'photo_weight': 0.6907244357800167, 'semantic_weight': 0.44523801361549925, 'temp': 0.9787292102992531}. Best is trial 12 with value: -0.8589819131485797.
# [I 2026-01-16 11:54:18,754] Trial 14 finished with value: -0.8590358715358716 and parameters: {'photo_weight': 0.8378209653288754, 'semantic_weight': 0.18288865337367177, 'temp': 0.8811758836363526}. Best is trial 14 with value: -0.8590358715358716.
# [I 2026-01-16 13:11:32,690] Trial 15 finished with value: -0.8579690371357037 and parameters: {'photo_weight': 0.8699890328757863, 'semantic_weight': 0.5131569665398066, 'temp': 0.8788410704185265}. Best is trial 14 with value: -0.8590358715358716.
# [I 2026-01-16 14:28:47,168] Trial 16 finished with value: -0.8590990799324132 and parameters: {'photo_weight': 0.7763975377410923, 'semantic_weight': 0.21527301417727468, 'temp': 0.8748815993499715}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-16 15:46:03,211] Trial 17 finished with value: -0.8570626903960237 and parameters: {'photo_weight': 0.7816359780099125, 'semantic_weight': 0.6031487154174974, 'temp': 0.8603847757802985}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-16 17:03:17,150] Trial 18 finished with value: -0.8588161838161839 and parameters: {'photo_weight': 0.78582979910311, 'semantic_weight': 0.4302912527830782, 'temp': 0.6326478006761422}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-16 18:20:31,019] Trial 19 finished with value: -0.858541921041921 and parameters: {'photo_weight': 0.8968132253979485, 'semantic_weight': 0.21008460001331974, 'temp': 0.9015353330808213}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-16 19:37:45,955] Trial 20 finished with value: -0.8568479360146027 and parameters: {'photo_weight': 0.5674342455054232, 'semantic_weight': 0.3975485332352269, 'temp': 0.8058110878946336}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-16 20:55:00,252] Trial 21 finished with value: -0.8586496836496836 and parameters: {'photo_weight': 0.7153241711329533, 'semantic_weight': 0.19966376627920068, 'temp': 0.9175219661456858}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-16 22:12:15,011] Trial 22 finished with value: -0.8583879083879085 and parameters: {'photo_weight': 0.8085566097829249, 'semantic_weight': 0.18355386487993242, 'temp': 0.9938401698354302}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-16 23:29:29,661] Trial 23 finished with value: -0.8588345296678629 and parameters: {'photo_weight': 0.6231945140851458, 'semantic_weight': 0.28028018713678887, 'temp': 0.8010344993973585}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 00:46:43,962] Trial 24 finished with value: -0.857990157990158 and parameters: {'photo_weight': 0.7227295594439116, 'semantic_weight': 0.16745129949455592, 'temp': 0.9340222743209087}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 02:03:58,566] Trial 25 finished with value: -0.8585018376685043 and parameters: {'photo_weight': 0.8282876560690495, 'semantic_weight': 0.2802468748001684, 'temp': 0.8475013386054857}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 03:21:13,185] Trial 26 finished with value: -0.8584571292904627 and parameters: {'photo_weight': 0.6048044936412433, 'semantic_weight': 0.24503574461343713, 'temp': 0.9438989371995872}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 04:38:26,824] Trial 27 finished with value: -0.8582258790592123 and parameters: {'photo_weight': 0.7263047675256509, 'semantic_weight': 0.37113036862614984, 'temp': 0.6474028406440107}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 05:55:41,741] Trial 28 finished with value: -0.8584011667345001 and parameters: {'photo_weight': 0.35589805887750997, 'semantic_weight': 0.15922832698689887, 'temp': 0.6908294639944054}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 07:12:57,810] Trial 29 finished with value: -0.8567495775829108 and parameters: {'photo_weight': 0.5335129551753739, 'semantic_weight': 0.4853250961301662, 'temp': 0.8169379136738213}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 08:30:13,201] Trial 30 finished with value: -0.8518845352178688 and parameters: {'photo_weight': 0.34855153575839726, 'semantic_weight': 0.665755582853081, 'temp': 0.8815654816170427}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 09:47:30,975] Trial 31 finished with value: -0.8583624708624707 and parameters: {'photo_weight': 0.667245361562024, 'semantic_weight': 0.10245843332982532, 'temp': 0.9590900920882005}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 11:04:46,699] Trial 32 finished with value: -0.8587967587967588 and parameters: {'photo_weight': 0.6462015365116743, 'semantic_weight': 0.16267816213198039, 'temp': 0.9294627795579176}. Best is trial 16 with value: -0.8590990799324132.
# [I 2026-01-17 12:22:02,889] Trial 33 finished with value: -0.8591354633021301 and parameters: {'photo_weight': 0.7511949644622653, 'semantic_weight': 0.25389140955540057, 'temp': 0.9990954833798346}. Best is trial 33 with value: -0.8591354633021301.
# [I 2026-01-17 13:39:18,370] Trial 34 finished with value: -0.8556327931327932 and parameters: {'photo_weight': 0.750161243976523, 'semantic_weight': 0.8721860143033557, 'temp': 0.9023916934614724}. Best is trial 33 with value: -0.8591354633021301.
# [I 2026-01-17 14:56:33,278] Trial 35 finished with value: -0.8594405594405594 and parameters: {'photo_weight': 0.8221257189722722, 'semantic_weight': 0.24503260111333713, 'temp': 0.9909817073404283}. Best is trial 35 with value: -0.8594405594405594.
# [I 2026-01-17 16:13:51,890] Trial 36 finished with value: -0.857532745032745 and parameters: {'photo_weight': 0.8371591566199763, 'semantic_weight': 0.2833244690047709, 'temp': 0.9947115722733336}. Best is trial 35 with value: -0.8594405594405594.
# [I 2026-01-17 17:31:08,477] Trial 37 finished with value: -0.8577615285948619 and parameters: {'photo_weight': 0.8948192662753555, 'semantic_weight': 0.2415769345899671, 'temp': 0.7598002371720215}. Best is trial 35 with value: -0.8594405594405594.
# [I 2026-01-17 18:48:25,611] Trial 38 finished with value: -0.8597266930600264 and parameters: {'photo_weight': 0.8234577040870251, 'semantic_weight': 0.3078856803172842, 'temp': 0.834220962125399}. Best is trial 38 with value: -0.8597266930600264.
# [I 2026-01-17 20:05:42,282] Trial 39 finished with value: -0.8581797831797833 and parameters: {'photo_weight': 0.7695875338046076, 'semantic_weight': 0.31703389605048715, 'temp': 0.7403725967548972}. Best is trial 38 with value: -0.8597266930600264.
# BEST BAYESIAN TRIAL
# Best params: {'photo_weight': 0.8234577040870251, 'semantic_weight': 0.3078856803172842, 'temp': 0.834220962125399}
# Best trial MRR: 0.8597266930600264
# Stored Hit Rate on best trial: 0.7851592851592851

# Evaluating full TRIAL set using best hyperparameters...

# ==============================
# FINAL RESULTS
# ==============================
# MRR:      0.752174
# Hit Rate: 0.617711
# Model_Path: /home/dbp52/Ambiguity-NLP/clip-vit-base-patch16







# ==============================
# Starting Bayesian tuning...   CLIP-ViT-B-32-laion2B-s34B-b79K
# ==============================

#   0%|          | 0/40 [00:00<?, ?it/s]/home/dbp52/Ambiguity-NLP/final.py:710: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   ctx = torch.cuda.amp.autocast(enabled=use_amp)

                                      

#   0%|          | 0/40 [1:01:49<?, ?it/s]
# Best trial: 0. Best value: -0.807112:   0%|          | 0/40 [1:01:49<?, ?it/s]
# Best trial: 0. Best value: -0.807112:   2%|▎         | 1/40 [1:01:49<40:11:25, 3709.88s/it]
                                                                                           

# Best trial: 0. Best value: -0.807112:   2%|▎         | 1/40 [2:03:37<40:11:25, 3709.88s/it]
# Best trial: 1. Best value: -0.824255:   2%|▎         | 1/40 [2:03:37<40:11:25, 3709.88s/it]
# Best trial: 1. Best value: -0.824255:   5%|▌         | 2/40 [2:03:37<39:08:41, 3708.47s/it]
                                                                                           

# Best trial: 1. Best value: -0.824255:   5%|▌         | 2/40 [3:05:29<39:08:41, 3708.47s/it]
# Best trial: 1. Best value: -0.824255:   5%|▌         | 2/40 [3:05:29<39:08:41, 3708.47s/it]
# Best trial: 1. Best value: -0.824255:   8%|▊         | 3/40 [3:05:29<38:07:52, 3710.06s/it]
                                                                                           

# Best trial: 1. Best value: -0.824255:   8%|▊         | 3/40 [4:07:12<38:07:52, 3710.06s/it]
# Best trial: 3. Best value: -0.825064:   8%|▊         | 3/40 [4:07:12<38:07:52, 3710.06s/it]
# Best trial: 3. Best value: -0.825064:  10%|█         | 4/40 [4:07:12<37:04:22, 3707.30s/it]
                                                                                           

# Best trial: 3. Best value: -0.825064:  10%|█         | 4/40 [5:08:19<37:04:22, 3707.30s/it]
# Best trial: 3. Best value: -0.825064:  10%|█         | 4/40 [5:08:19<37:04:22, 3707.30s/it]
# Best trial: 3. Best value: -0.825064:  12%|█▎        | 5/40 [5:08:19<35:54:09, 3692.83s/it]
                                                                                           

# Best trial: 3. Best value: -0.825064:  12%|█▎        | 5/40 [6:09:43<35:54:09, 3692.83s/it]
# Best trial: 3. Best value: -0.825064:  12%|█▎        | 5/40 [6:09:43<35:54:09, 3692.83s/it]
# Best trial: 3. Best value: -0.825064:  15%|█▌        | 6/40 [6:09:43<34:50:49, 3689.70s/it]
                                                                                           

# Best trial: 3. Best value: -0.825064:  15%|█▌        | 6/40 [7:11:05<34:50:49, 3689.70s/it]
# Best trial: 3. Best value: -0.825064:  15%|█▌        | 6/40 [7:11:05<34:50:49, 3689.70s/it]
# Best trial: 3. Best value: -0.825064:  18%|█▊        | 7/40 [7:11:05<33:48:00, 3687.29s/it]
                                                                                           

# Best trial: 3. Best value: -0.825064:  18%|█▊        | 7/40 [8:12:38<33:48:00, 3687.29s/it]
# Best trial: 3. Best value: -0.825064:  18%|█▊        | 7/40 [8:12:38<33:48:00, 3687.29s/it]
# Best trial: 3. Best value: -0.825064:  20%|██        | 8/40 [8:12:38<32:47:34, 3689.21s/it]
                                                                                           

# Best trial: 3. Best value: -0.825064:  20%|██        | 8/40 [9:14:13<32:47:34, 3689.21s/it]
# Best trial: 3. Best value: -0.825064:  20%|██        | 8/40 [9:14:13<32:47:34, 3689.21s/it]
# Best trial: 3. Best value: -0.825064:  22%|██▎       | 9/40 [9:14:13<31:46:59, 3690.96s/it]
                                                                                           

# Best trial: 3. Best value: -0.825064:  22%|██▎       | 9/40 [10:15:53<31:46:59, 3690.96s/it]
# Best trial: 3. Best value: -0.825064:  22%|██▎       | 9/40 [10:15:53<31:46:59, 3690.96s/it]
# Best trial: 3. Best value: -0.825064:  25%|██▌       | 10/40 [10:15:53<30:46:50, 3693.70s/it]
                                                                                             

# Best trial: 3. Best value: -0.825064:  25%|██▌       | 10/40 [11:17:25<30:46:50, 3693.70s/it]
# Best trial: 10. Best value: -0.826416:  25%|██▌       | 10/40 [11:17:25<30:46:50, 3693.70s/it]
# Best trial: 10. Best value: -0.826416:  28%|██▊       | 11/40 [11:17:25<29:45:01, 3693.17s/it]
                                                                                              

# Best trial: 10. Best value: -0.826416:  28%|██▊       | 11/40 [12:18:55<29:45:01, 3693.17s/it]
# Best trial: 10. Best value: -0.826416:  28%|██▊       | 11/40 [12:18:55<29:45:01, 3693.17s/it]
# Best trial: 10. Best value: -0.826416:  30%|███       | 12/40 [12:18:55<28:43:06, 3692.37s/it]
                                                                                              

# Best trial: 10. Best value: -0.826416:  30%|███       | 12/40 [13:20:26<28:43:06, 3692.37s/it]
# Best trial: 12. Best value: -0.826658:  30%|███       | 12/40 [13:20:26<28:43:06, 3692.37s/it]
# Best trial: 12. Best value: -0.826658:  32%|███▎      | 13/40 [13:20:26<27:41:16, 3691.71s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  32%|███▎      | 13/40 [14:21:59<27:41:16, 3691.71s/it]
# Best trial: 12. Best value: -0.826658:  32%|███▎      | 13/40 [14:21:59<27:41:16, 3691.71s/it]
# Best trial: 12. Best value: -0.826658:  35%|███▌      | 14/40 [14:21:59<26:39:55, 3692.14s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  35%|███▌      | 14/40 [15:23:14<26:39:55, 3692.14s/it]
# Best trial: 12. Best value: -0.826658:  35%|███▌      | 14/40 [15:23:14<26:39:55, 3692.14s/it]
# Best trial: 12. Best value: -0.826658:  38%|███▊      | 15/40 [15:23:14<25:36:13, 3686.93s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  38%|███▊      | 15/40 [16:24:45<25:36:13, 3686.93s/it]
# Best trial: 12. Best value: -0.826658:  38%|███▊      | 15/40 [16:24:45<25:36:13, 3686.93s/it]
# Best trial: 12. Best value: -0.826658:  40%|████      | 16/40 [16:24:45<24:35:21, 3688.40s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  40%|████      | 16/40 [17:26:13<24:35:21, 3688.40s/it]
# Best trial: 12. Best value: -0.826658:  40%|████      | 16/40 [17:26:13<24:35:21, 3688.40s/it]
# Best trial: 12. Best value: -0.826658:  42%|████▎     | 17/40 [17:26:13<23:33:44, 3688.02s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  42%|████▎     | 17/40 [18:27:40<23:33:44, 3688.02s/it]
# Best trial: 12. Best value: -0.826658:  42%|████▎     | 17/40 [18:27:40<23:33:44, 3688.02s/it]
# Best trial: 12. Best value: -0.826658:  45%|████▌     | 18/40 [18:27:40<22:32:15, 3687.98s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  45%|████▌     | 18/40 [19:29:09<22:32:15, 3687.98s/it]
# Best trial: 12. Best value: -0.826658:  45%|████▌     | 18/40 [19:29:09<22:32:15, 3687.98s/it]
# Best trial: 12. Best value: -0.826658:  48%|████▊     | 19/40 [19:29:09<21:30:50, 3688.10s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  48%|████▊     | 19/40 [20:30:39<21:30:50, 3688.10s/it]
# Best trial: 12. Best value: -0.826658:  48%|████▊     | 19/40 [20:30:39<21:30:50, 3688.10s/it]
# Best trial: 12. Best value: -0.826658:  50%|█████     | 20/40 [20:30:39<20:29:32, 3688.64s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  50%|█████     | 20/40 [21:32:16<20:29:32, 3688.64s/it]
# Best trial: 12. Best value: -0.826658:  50%|█████     | 20/40 [21:32:16<20:29:32, 3688.64s/it]
# Best trial: 12. Best value: -0.826658:  52%|█████▎    | 21/40 [21:32:16<19:28:52, 3691.21s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  52%|█████▎    | 21/40 [22:33:51<19:28:52, 3691.21s/it]
# Best trial: 12. Best value: -0.826658:  52%|█████▎    | 21/40 [22:33:51<19:28:52, 3691.21s/it]
# Best trial: 12. Best value: -0.826658:  55%|█████▌    | 22/40 [22:33:51<18:27:41, 3692.29s/it]
                                                                                              

# Best trial: 12. Best value: -0.826658:  55%|█████▌    | 22/40 [23:35:26<18:27:41, 3692.29s/it]
# Best trial: 22. Best value: -0.826707:  55%|█████▌    | 22/40 [23:35:26<18:27:41, 3692.29s/it]
# Best trial: 22. Best value: -0.826707:  57%|█████▊    | 23/40 [23:35:26<17:26:25, 3693.25s/it]
                                                                                              

# Best trial: 22. Best value: -0.826707:  57%|█████▊    | 23/40 [24:37:02<17:26:25, 3693.25s/it]
# Best trial: 22. Best value: -0.826707:  57%|█████▊    | 23/40 [24:37:02<17:26:25, 3693.25s/it]
# Best trial: 22. Best value: -0.826707:  60%|██████    | 24/40 [24:37:02<16:25:02, 3693.90s/it]
                                                                                              

# Best trial: 22. Best value: -0.826707:  60%|██████    | 24/40 [25:38:34<16:25:02, 3693.90s/it]
# Best trial: 22. Best value: -0.826707:  60%|██████    | 24/40 [25:38:34<16:25:02, 3693.90s/it]
# Best trial: 22. Best value: -0.826707:  62%|██████▎   | 25/40 [25:38:34<15:23:19, 3693.32s/it]
                                                                                              

# Best trial: 22. Best value: -0.826707:  62%|██████▎   | 25/40 [26:40:12<15:23:19, 3693.32s/it]
# Best trial: 22. Best value: -0.826707:  62%|██████▎   | 25/40 [26:40:12<15:23:19, 3693.32s/it]
# Best trial: 22. Best value: -0.826707:  65%|██████▌   | 26/40 [26:40:12<14:22:08, 3694.86s/it]
                                                                                              

# Best trial: 22. Best value: -0.826707:  65%|██████▌   | 26/40 [27:41:45<14:22:08, 3694.86s/it]
# Best trial: 22. Best value: -0.826707:  65%|██████▌   | 26/40 [27:41:45<14:22:08, 3694.86s/it]
# Best trial: 22. Best value: -0.826707:  68%|██████▊   | 27/40 [27:41:45<13:20:26, 3694.31s/it]
                                                                                              

# Best trial: 22. Best value: -0.826707:  68%|██████▊   | 27/40 [28:43:26<13:20:26, 3694.31s/it]
# Best trial: 27. Best value: -0.82745:  68%|██████▊   | 27/40 [28:43:26<13:20:26, 3694.31s/it] 
# Best trial: 27. Best value: -0.82745:  70%|███████   | 28/40 [28:43:26<12:19:14, 3696.20s/it]
                                                                                             

# Best trial: 27. Best value: -0.82745:  70%|███████   | 28/40 [29:45:02<12:19:14, 3696.20s/it]
# Best trial: 27. Best value: -0.82745:  70%|███████   | 28/40 [29:45:02<12:19:14, 3696.20s/it]
# Best trial: 27. Best value: -0.82745:  72%|███████▎  | 29/40 [29:45:02<11:17:39, 3696.31s/it]
                                                                                             

# Best trial: 27. Best value: -0.82745:  72%|███████▎  | 29/40 [30:46:21<11:17:39, 3696.31s/it]
# Best trial: 27. Best value: -0.82745:  72%|███████▎  | 29/40 [30:46:21<11:17:39, 3696.31s/it]
# Best trial: 27. Best value: -0.82745:  75%|███████▌  | 30/40 [30:46:21<10:15:11, 3691.16s/it]
                                                                                             

# Best trial: 27. Best value: -0.82745:  75%|███████▌  | 30/40 [31:47:43<10:15:11, 3691.16s/it]
# Best trial: 27. Best value: -0.82745:  75%|███████▌  | 30/40 [31:47:43<10:15:11, 3691.16s/it]
# Best trial: 27. Best value: -0.82745:  78%|███████▊  | 31/40 [31:47:43<9:13:14, 3688.23s/it] 
                                                                                            

# Best trial: 27. Best value: -0.82745:  78%|███████▊  | 31/40 [32:48:53<9:13:14, 3688.23s/it]
# Best trial: 27. Best value: -0.82745:  78%|███████▊  | 31/40 [32:48:53<9:13:14, 3688.23s/it]
# Best trial: 27. Best value: -0.82745:  80%|████████  | 32/40 [32:48:53<8:11:02, 3682.84s/it]
                                                                                            

# Best trial: 27. Best value: -0.82745:  80%|████████  | 32/40 [33:50:21<8:11:02, 3682.84s/it]
# Best trial: 27. Best value: -0.82745:  80%|████████  | 32/40 [33:50:21<8:11:02, 3682.84s/it]
# Best trial: 27. Best value: -0.82745:  82%|████████▎ | 33/40 [33:50:21<7:09:50, 3684.29s/it]
                                                                                            

# Best trial: 27. Best value: -0.82745:  82%|████████▎ | 33/40 [34:51:46<7:09:50, 3684.29s/it]
# Best trial: 27. Best value: -0.82745:  82%|████████▎ | 33/40 [34:51:46<7:09:50, 3684.29s/it]
# Best trial: 27. Best value: -0.82745:  85%|████████▌ | 34/40 [34:51:46<6:08:27, 3684.59s/it]
                                                                                            

# Best trial: 27. Best value: -0.82745:  85%|████████▌ | 34/40 [35:52:55<6:08:27, 3684.59s/it]
# Best trial: 27. Best value: -0.82745:  85%|████████▌ | 34/40 [35:52:55<6:08:27, 3684.59s/it]
# Best trial: 27. Best value: -0.82745:  88%|████████▊ | 35/40 [35:52:55<5:06:38, 3679.77s/it]
                                                                                            

# Best trial: 27. Best value: -0.82745:  88%|████████▊ | 35/40 [36:54:17<5:06:38, 3679.77s/it]
# Best trial: 27. Best value: -0.82745:  88%|████████▊ | 35/40 [36:54:17<5:06:38, 3679.77s/it]
# Best trial: 27. Best value: -0.82745:  90%|█████████ | 36/40 [36:54:17<4:05:21, 3680.50s/it]
                                                                                            

# Best trial: 27. Best value: -0.82745:  90%|█████████ | 36/40 [37:55:27<4:05:21, 3680.50s/it]
# Best trial: 27. Best value: -0.82745:  90%|█████████ | 36/40 [37:55:27<4:05:21, 3680.50s/it]
# Best trial: 27. Best value: -0.82745:  92%|█████████▎| 37/40 [37:55:27<3:03:52, 3677.53s/it]
                                                                                            

# Best trial: 27. Best value: -0.82745:  92%|█████████▎| 37/40 [38:56:37<3:03:52, 3677.53s/it]
# Best trial: 27. Best value: -0.82745:  92%|█████████▎| 37/40 [38:56:37<3:03:52, 3677.53s/it]
# Best trial: 27. Best value: -0.82745:  95%|█████████▌| 38/40 [38:56:37<2:02:30, 3675.22s/it]
                                                                                            

# Best trial: 27. Best value: -0.82745:  95%|█████████▌| 38/40 [39:57:49<2:02:30, 3675.22s/it]
# Best trial: 27. Best value: -0.82745:  95%|█████████▌| 38/40 [39:57:49<2:02:30, 3675.22s/it]
# Best trial: 27. Best value: -0.82745:  98%|█████████▊| 39/40 [39:57:49<1:01:14, 3674.19s/it]
                                                                                            

# Best trial: 27. Best value: -0.82745:  98%|█████████▊| 39/40 [40:59:00<1:01:14, 3674.19s/it]
# Best trial: 27. Best value: -0.82745:  98%|█████████▊| 39/40 [40:59:00<1:01:14, 3674.19s/it]
# Best trial: 27. Best value: -0.82745: 100%|██████████| 40/40 [40:59:00<00:00, 3673.13s/it]  
# Best trial: 27. Best value: -0.82745: 100%|██████████| 40/40 [40:59:00<00:00, 3688.51s/it]
# [I 2026-01-15 17:41:26,435] Trial 0 finished with value: -0.8071124862791529 and parameters: {'photo_weight': 0.39963209507789, 'semantic_weight': 0.8605714451279329, 'temp': 0.839196365086843}. Best is trial 0 with value: -0.8071124862791529.
# [I 2026-01-15 18:43:13,923] Trial 1 finished with value: -0.8242547575880909 and parameters: {'photo_weight': 0.5789267873576293, 'semantic_weight': 0.22481491235394924, 'temp': 0.49359671220172163}. Best is trial 1 with value: -0.8242547575880909.
# [I 2026-01-15 19:45:05,867] Trial 2 finished with value: -0.8001570959904294 and parameters: {'photo_weight': 0.14646688973455957, 'semantic_weight': 0.7929409166199481, 'temp': 0.7606690070459252}. Best is trial 1 with value: -0.8242547575880909.
# [I 2026-01-15 20:46:48,929] Trial 3 finished with value: -0.8250639792306458 and parameters: {'photo_weight': 0.6664580622368363, 'semantic_weight': 0.11646759543664197, 'temp': 0.9819459112971965}. Best is trial 3 with value: -0.8250639792306458.
# [I 2026-01-15 21:47:56,120] Trial 4 finished with value: -0.824548137048137 and parameters: {'photo_weight': 0.7659541126403374, 'semantic_weight': 0.26987128854262094, 'temp': 0.5090949803242604}. Best is trial 3 with value: -0.8250639792306458.
# [I 2026-01-15 22:49:19,750] Trial 5 finished with value: -0.8131231422898089 and parameters: {'photo_weight': 0.24672360788274705, 'semantic_weight': 0.3433937943676302, 'temp': 0.7148538589793427}. Best is trial 3 with value: -0.8250639792306458.
# [I 2026-01-15 23:50:42,080] Trial 6 finished with value: -0.8179064762398095 and parameters: {'photo_weight': 0.4455560149136927, 'semantic_weight': 0.3329833121584336, 'temp': 0.7671117368334277}. Best is trial 3 with value: -0.8250639792306458.
# [I 2026-01-16 00:52:15,392] Trial 7 finished with value: -0.8108467150133817 and parameters: {'photo_weight': 0.21159508852163347, 'semantic_weight': 0.3337157188281745, 'temp': 0.619817105976215}. Best is trial 3 with value: -0.8250639792306458.
# [I 2026-01-16 01:53:50,210] Trial 8 finished with value: -0.8116269532936199 and parameters: {'photo_weight': 0.46485598737362877, 'semantic_weight': 0.728140769114411, 'temp': 0.5198042692950159}. Best is trial 3 with value: -0.8250639792306458.
# [I 2026-01-16 02:55:30,033] Trial 9 finished with value: -0.8146993438660105 and parameters: {'photo_weight': 0.5113875507308893, 'semantic_weight': 0.573931655089634, 'temp': 0.42787024763199866}. Best is trial 3 with value: -0.8250639792306458.
# [I 2026-01-16 03:57:01,989] Trial 10 finished with value: -0.8264160222493555 and parameters: {'photo_weight': 0.8743504239092522, 'semantic_weight': 0.10479774636465367, 'temp': 0.9817222664727197}. Best is trial 10 with value: -0.8264160222493555.
# [I 2026-01-16 04:58:32,551] Trial 11 finished with value: -0.8261995720329053 and parameters: {'photo_weight': 0.8475706336957293, 'semantic_weight': 0.11161368127068712, 'temp': 0.9445988771695106}. Best is trial 10 with value: -0.8264160222493555.
# [I 2026-01-16 06:00:02,739] Trial 12 finished with value: -0.8266583724917058 and parameters: {'photo_weight': 0.8874723746984159, 'semantic_weight': 0.1046550896560175, 'temp': 0.9896608556283883}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 07:01:35,859] Trial 13 finished with value: -0.8211782045115379 and parameters: {'photo_weight': 0.8981454808967397, 'semantic_weight': 0.4946173099368798, 'temp': 0.8682768980047672}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 08:02:50,727] Trial 14 finished with value: -0.825383258716592 and parameters: {'photo_weight': 0.7386173648317147, 'semantic_weight': 0.18987174408992893, 'temp': 0.9138372632416132}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 09:04:22,543] Trial 15 finished with value: -0.8206540373207041 and parameters: {'photo_weight': 0.7970181829268084, 'semantic_weight': 0.4596788067206572, 'temp': 0.9993787386969284}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 10:05:49,693] Trial 16 finished with value: -0.8173318964985632 and parameters: {'photo_weight': 0.6599965620925751, 'semantic_weight': 0.6166851525625929, 'temp': 0.858215162861321}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 11:07:17,554] Trial 17 finished with value: -0.822329830663164 and parameters: {'photo_weight': 0.8990129600955228, 'semantic_weight': 0.4129611095260306, 'temp': 0.6264965712313106}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 12:08:45,950] Trial 18 finished with value: -0.8243619035285702 and parameters: {'photo_weight': 0.6609300763150079, 'semantic_weight': 0.18936207511839553, 'temp': 0.9107343340254438}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 13:10:15,858] Trial 19 finished with value: -0.8240791615791616 and parameters: {'photo_weight': 0.374239430613053, 'semantic_weight': 0.13351227861964957, 'temp': 0.8028625330990015}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 14:11:53,044] Trial 20 finished with value: -0.823772215438882 and parameters: {'photo_weight': 0.7287014174903133, 'semantic_weight': 0.27407316579043295, 'temp': 0.9327247695317423}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 15:13:27,868] Trial 21 finished with value: -0.825518463018463 and parameters: {'photo_weight': 0.841998827272159, 'semantic_weight': 0.12393946410322784, 'temp': 0.9645687038832438}. Best is trial 12 with value: -0.8266583724917058.
# [I 2026-01-16 16:15:03,354] Trial 22 finished with value: -0.8267069350402684 and parameters: {'photo_weight': 0.8543530596821339, 'semantic_weight': 0.10429823900128356, 'temp': 0.9206862268684902}. Best is trial 22 with value: -0.8267069350402684.
# [I 2026-01-16 17:16:38,772] Trial 23 finished with value: -0.8259738718072052 and parameters: {'photo_weight': 0.8058540837832704, 'semantic_weight': 0.19476292041861892, 'temp': 0.8932312810996659}. Best is trial 22 with value: -0.8267069350402684.
# [I 2026-01-16 18:18:10,737] Trial 24 finished with value: -0.8234295025961692 and parameters: {'photo_weight': 0.5846573393763789, 'semantic_weight': 0.25502075993512086, 'temp': 0.9760153901899322}. Best is trial 22 with value: -0.8267069350402684.
# [I 2026-01-16 19:19:49,203] Trial 25 finished with value: -0.8224405224405223 and parameters: {'photo_weight': 0.8546890420475233, 'semantic_weight': 0.3983226442007181, 'temp': 0.8252748459161005}. Best is trial 22 with value: -0.8267069350402684.
# [I 2026-01-16 20:21:22,232] Trial 26 finished with value: -0.8248970165636832 and parameters: {'photo_weight': 0.7051975453153504, 'semantic_weight': 0.16410612598094382, 'temp': 0.992902717487556}. Best is trial 22 with value: -0.8267069350402684.
# [I 2026-01-16 21:23:02,850] Trial 27 finished with value: -0.8274501732835066 and parameters: {'photo_weight': 0.7957995374655943, 'semantic_weight': 0.1042386719269501, 'temp': 0.8918996747187171}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-16 22:24:39,421] Trial 28 finished with value: -0.8243763952097285 and parameters: {'photo_weight': 0.7865511511871359, 'semantic_weight': 0.2387706024878012, 'temp': 0.6758027690459208}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-16 23:25:58,545] Trial 29 finished with value: -0.816218195384862 and parameters: {'photo_weight': 0.6007983398611744, 'semantic_weight': 0.6034629328238524, 'temp': 0.8664271662327774}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 00:27:19,959] Trial 30 finished with value: -0.8066911483578151 and parameters: {'photo_weight': 0.3477606237733729, 'semantic_weight': 0.8545880924671956, 'temp': 0.7999213835846314}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 01:28:30,218] Trial 31 finished with value: -0.8264258889258889 and parameters: {'photo_weight': 0.8369151512141751, 'semantic_weight': 0.10142661075986524, 'temp': 0.9266908925831108}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 02:29:57,888] Trial 32 finished with value: -0.8252542210875545 and parameters: {'photo_weight': 0.8279625779077167, 'semantic_weight': 0.16969825416516277, 'temp': 0.9017578714244637}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 03:31:23,162] Trial 33 finished with value: -0.8242700201033534 and parameters: {'photo_weight': 0.7511949644622653, 'semantic_weight': 0.22970732335854815, 'temp': 0.9447041541084836}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 04:32:31,683] Trial 34 finished with value: -0.8256168214501547 and parameters: {'photo_weight': 0.6952740492017327, 'semantic_weight': 0.15357311130702045, 'temp': 0.8234322899414998}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 05:33:53,895] Trial 35 finished with value: -0.8241682700016033 and parameters: {'photo_weight': 0.802210416019149, 'semantic_weight': 0.29580042715430777, 'temp': 0.8824739680229527}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 06:35:04,491] Trial 36 finished with value: -0.8242783451116784 and parameters: {'photo_weight': 0.6223077354447096, 'semantic_weight': 0.18574949689970133, 'temp': 0.7658387755364872}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 07:36:14,336] Trial 37 finished with value: -0.8263659180325846 and parameters: {'photo_weight': 0.7576478962392191, 'semantic_weight': 0.10190490277323444, 'temp': 0.9327664373633167}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 08:37:26,103] Trial 38 finished with value: -0.8235149110149109 and parameters: {'photo_weight': 0.5280712394979159, 'semantic_weight': 0.21304976637551148, 'temp': 0.8365458466708997}. Best is trial 27 with value: -0.8274501732835066.
# [I 2026-01-17 09:38:36,769] Trial 39 finished with value: -0.8232195273861941 and parameters: {'photo_weight': 0.8544523339901563, 'semantic_weight': 0.31001372197586197, 'temp': 0.9486845605123937}. Best is trial 27 with value: -0.8274501732835066.
# BEST BAYESIAN TRIAL
# Best params: {'photo_weight': 0.7957995374655943, 'semantic_weight': 0.1042386719269501, 'temp': 0.8918996747187171}
# Best trial MRR: 0.8274501732835066
# Stored Hit Rate on best trial: 0.7435897435897436

# Evaluating full TRIAL set using best hyperparameters...

# ==============================
# FINAL RESULTS
# ==============================
# MRR:      0.759022
# Hit Rate: 0.622030
# Model_Path: /home/dbp52/Ambiguity-NLP/CLIP-ViT-B-32-laion2B-s34B-b79K