# core.py
import os
import random
import pickle
from typing import Literal, Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

import nltk
from nltk.corpus import wordnet as wn
from transformers import CLIPModel, CLIPProcessor

import time

# -----------------------------
# Data loading (same as your code)
# -----------------------------
def load_data(file_path, train_val="trial", target_size=(384, 384), use_cache=True):
    path = os.path.join(file_path, train_val + "_v1")
    cache_file = os.path.join(path, "image_cache.pkl")

    path_data = os.path.join(path, train_val + ".data.v1.txt")
    data = pd.read_csv(path_data, sep="\t", header=None)
    data.columns = ["target", "sentence"] + [f"image_{i}" for i in range(data.shape[1] - 2)]

    path_labels = os.path.join(path, train_val + ".gold.v1.txt")
    with open(path_labels, "r", encoding="utf-8") as f:
        gold_labels = [line.strip() for line in f]
    data["label"] = gold_labels

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached images from {cache_file}...")
        with open(cache_file, "rb") as f:
            image_dict = pickle.load(f)
        print(f"Loaded {len(image_dict)} cached images")
        return data, image_dict

    path_images = os.path.join(path, train_val + "_images_v1")
    image_dict = {}
    files = os.listdir(path_images)
    for filename in tqdm(files, total=len(files), desc="Loading Images", unit="image"):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            try:
                img = Image.open(os.path.join(path_images, filename)).convert("RGB")
                img_resized = img.resize(target_size, resample=Image.BICUBIC)
                image_dict[filename] = img_resized
            except Exception:
                continue

    if use_cache:
        print(f"Saving images to cache: {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(image_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cached {len(image_dict)} images")

    return data, image_dict


# -----------------------------
# Prompting helpers (same logic as your code)
# -----------------------------
def extract_context_word(target: str, sentence: str) -> str:
    if sentence is None:
        return ""
    s = str(sentence).strip()
    t = str(target).strip()
    if not s:
        return ""
    tokens = s.split()
    if len(tokens) == 2:
        t_low = t.lower()
        tok0, tok1 = tokens[0], tokens[1]
        if tok0.lower() == t_low:
            return tok1
        if tok1.lower() == t_low:
            return tok0
    return s


def get_synonym(word: str) -> Optional[str]:
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


def build_photo_prompts(target: str, sentence: str) -> List[str]:
    target = str(target).strip()
    sentence = "" if sentence is None else str(sentence).strip()

    context = extract_context_word(target, sentence).strip()
    base_context = context if context else sentence

    prompts = [
        f"a photo of {target} {base_context}".strip(),
        f"{target} {base_context}, realistic photo".strip(),
        f"{target} near {base_context}, real world".strip(),
        f"{target} with {base_context}, natural scene".strip(),
        f"{target} appearing in a {base_context} environment".strip(),
        f"{target} {base_context}, high quality photograph".strip(),
    ]

    t_syn = get_synonym(target)
    c_syn = get_synonym(context) if context else None
    syn_target = t_syn if t_syn else target
    syn_context = c_syn if c_syn else base_context
    if syn_target and syn_context:
        prompts.append(f"a photo of {syn_target} {syn_context}".strip())

    return [p for p in prompts if p]


def build_semantic_prompts(target: str, sentence: str) -> List[str]:
    target = str(target).strip()
    sentence = "" if sentence is None else str(sentence).strip()

    context = extract_context_word(target, sentence).strip()
    base_context = context if context else sentence

    prompts = [
        f"{target} related to {base_context}".strip(),
        f"the concept of {target} in {base_context}".strip(),
        f"{target} in the context of {base_context}".strip(),
        f"visual sense of {target} in {base_context}".strip(),
        f"illustration of {target} used with {base_context}".strip(),
    ]

    if sentence:
        prompts.extend([
            f"a visual depiction of: {sentence}".strip(),
            f"illustration of the phrase: {sentence}".strip(),
            f"{sentence} (image)".strip(),
            f"{sentence}, realistic photography".strip(),
        ])

    return [p for p in prompts if p]


def get_prompted_text_embedding(prompts: List[str], processor: CLIPProcessor, model: CLIPModel) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    if len(prompts) == 0:
        d = model.config.projection_dim
        return torch.zeros(d, device=device)

    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)
        feats = F.normalize(feats, p=2, dim=-1)
        mean_feat = feats.mean(dim=0)
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
    photo_prompts = build_photo_prompts(target, sentence)
    sem_prompts = build_semantic_prompts(target, sentence)

    photo_emb = get_prompted_text_embedding(photo_prompts, processor, model)
    sem_emb = get_prompted_text_embedding(sem_prompts, processor, model)

    combined = photo_weight * photo_emb + semantic_weight * sem_emb
    combined = F.normalize(combined, p=2, dim=-1)
    combined = F.normalize(combined, p=2, dim=-1)
    return combined


def get_plain_text_embedding(target: str, sentence: str, processor: CLIPProcessor, model: CLIPModel) -> torch.Tensor:
    """
    For 'augmentations only' baseline:
    - no prompt ensembles
    - just encode the raw sentence once (or "target sentence")
    """
    model.eval()
    device = next(model.parameters()).device
    text = f"{target} {sentence}".strip()

    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feat = model.get_text_features(**inputs).squeeze(0)
        feat = F.normalize(feat, p=2, dim=-1)
    return feat


# -----------------------------
# Image augmentations (same as your code)
# -----------------------------
def random_geometric_augment(img: Image.Image) -> Image.Image:
    w, h = img.size
    out = img
    if random.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    angle = random.uniform(-7, 7)
    out = out.rotate(angle, resample=Image.BICUBIC, expand=False)

    scale = random.uniform(0.95, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w < w and new_h < h:
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        out = out.crop((left, top, left + new_w, top + new_h))
        out = out.resize((w, h), resample=Image.BICUBIC)
    return out


def random_photometric_augment(img: Image.Image) -> Image.Image:
    out = img
    if random.random() < 0.5:
        out = ImageEnhance.Brightness(out).enhance(random.uniform(0.95, 1.05))
    if random.random() < 0.5:
        out = ImageEnhance.Contrast(out).enhance(random.uniform(0.95, 1.05))
    if random.random() < 0.5:
        out = ImageEnhance.Color(out).enhance(random.uniform(0.95, 1.05))
    if random.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=0.3))
    return out


def generate_tta_views(img: Image.Image, num_random_augs: int = 2, out_size=(224, 224)) -> List[Image.Image]:
    views = []
    w, h = img.size

    views.append(img.resize(out_size, resample=Image.BICUBIC))  # base
    views.append(img.transpose(Image.FLIP_LEFT_RIGHT).resize(out_size, resample=Image.BICUBIC))  # flip

    crop_scale = 0.8
    cw, ch = int(w * crop_scale), int(h * crop_scale)
    left = (w - cw) // 2
    top = (h - ch) // 2
    center_crop = img.crop((left, top, left + cw, top + ch)).resize(out_size, resample=Image.BICUBIC)
    views.append(center_crop)

    zoom_scale = 0.6
    zw, zh = int(w * zoom_scale), int(h * zoom_scale)
    zleft = (w - zw) // 2
    ztop = (h - zh) // 2
    zoom_crop = img.crop((zleft, ztop, zleft + zw, ztop + zh)).resize(out_size, resample=Image.BICUBIC)
    views.append(zoom_crop)

    gray = img.convert("L").convert("RGB").resize(out_size, resample=Image.BICUBIC)
    views.append(gray)

    for _ in range(num_random_augs):
        aug = random_geometric_augment(img)
        aug = random_photometric_augment(aug)
        views.append(aug.resize(out_size, resample=Image.BICUBIC))

    return views


def multi_crops(img: Image.Image, out_size=(224, 224)) -> List[Image.Image]:
    w, h = img.size
    crops = []

    crop_w, crop_h = max(1, int(w * 0.8)), max(1, int(h * 0.8))

    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    crops.append(img.crop((left, top, left + crop_w, top + crop_h)).resize(out_size, resample=Image.BICUBIC))

    crops.append(img.crop((0, 0, crop_w, crop_h)).resize(out_size, resample=Image.BICUBIC))
    crops.append(img.crop((w - crop_w, 0, w, crop_h)).resize(out_size, resample=Image.BICUBIC))
    crops.append(img.crop((0, h - crop_h, crop_w, h)).resize(out_size, resample=Image.BICUBIC))
    crops.append(img.crop((w - crop_w, h - crop_h, w, h)).resize(out_size, resample=Image.BICUBIC))

    zoom_scale = 0.6
    z_w, z_h = int(w * zoom_scale), int(h * zoom_scale)
    z_left = (w - z_w) // 2
    z_top = (h - z_h) // 2
    crops.append(img.crop((z_left, z_top, z_left + z_w, z_top + z_h)).resize(out_size, resample=Image.BICUBIC))

    return crops


def grid_patches(img: Image.Image, grid_size=3, out_size=(224, 224)) -> List[Image.Image]:
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
            patches.append(img.crop((left, top, right, bottom)).resize(out_size, resample=Image.BICUBIC))
    return patches


def center_saliency_crops(img: Image.Image, out_size=(224, 224)) -> List[Image.Image]:
    w, h = img.size
    crops = []
    for scale in [0.7, 0.5]:
        cw, ch = int(w * scale), int(h * scale)
        left = (w - cw) // 2
        top = (h - ch) // 2
        crops.append(img.crop((left, top, left + cw, top + ch)).resize(out_size, resample=Image.BICUBIC))
    return crops


def mid_quadrant_crops(img: Image.Image, out_size=(224, 224)) -> List[Image.Image]:
    w, h = img.size
    crops = []
    cw, ch = w // 2, h // 2
    centers = [(w // 4, h // 4), (3 * w // 4, h // 4), (w // 4, 3 * h // 4), (3 * w // 4, 3 * h // 4)]
    for cx, cy in centers:
        left = max(0, cx - cw // 2)
        top = max(0, cy - ch // 2)
        right = min(w, left + cw)
        bottom = min(h, top + ch)
        crops.append(img.crop((left, top, right, bottom)).resize(out_size, resample=Image.BICUBIC))
    return crops


def get_image_embedding_multi_view(
    img: Image.Image,
    processor: CLIPProcessor,
    model: CLIPModel,
    temp: float = 0.7,
    out_size=(224, 224),
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device

    views: List[Image.Image] = []
    views.extend(generate_tta_views(img, num_random_augs=2, out_size=out_size))
    views.extend(multi_crops(img, out_size=out_size))
    views.extend(grid_patches(img, grid_size=3, out_size=out_size))
    views.extend(center_saliency_crops(img, out_size=out_size))
    views.extend(mid_quadrant_crops(img, out_size=out_size))

    with torch.no_grad():
        inputs = processor(images=views, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, p=2, dim=-1)
        feat_mean = feats.mean(dim=0)
        feat_scaled = feat_mean / temp
        feat_final = F.normalize(feat_scaled, p=2, dim=-1)
    return feat_final


def get_image_embedding_single_view(
    img: Image.Image,
    processor: CLIPProcessor,
    model: CLIPModel,
    temp: float = 0.7,
    out_size=(224, 224),
) -> torch.Tensor:
    """
    For 'prompting only' baseline:
    - no multi-view crops/augs
    - just one resized image into CLIP
    """
    model.eval()
    device = next(model.parameters()).device
    one = img.resize(out_size, resample=Image.BICUBIC)
    with torch.no_grad():
        inputs = processor(images=[one], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feat = model.get_image_features(**inputs).squeeze(0)
        feat = F.normalize(feat, p=2, dim=-1)
        feat = F.normalize((feat / temp), p=2, dim=-1)
    return feat


# -----------------------------
# Configurable choose_image + eval
# -----------------------------
def choose_image_configurable(
    target: str,
    sentence: str,
    images: List[str],
    image_dict: Dict[str, Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
    *,
    use_prompting: bool,
    use_augmentations: bool,
    photo_weight: float = 0.6,
    semantic_weight: float = 0.4,
    temp: float = 0.7,
) -> List[str]:
    """
    This is the key: one choose_image that can emulate:
      - only augmentations
      - only prompting
      - both
    """

    if use_prompting:
        text_emb = get_dual_channel_text_embedding(
            target=target,
            sentence=sentence,
            processor=processor,
            model=model,
            photo_weight=photo_weight,
            semantic_weight=semantic_weight,
        )
    else:
        text_emb = get_plain_text_embedding(target, sentence, processor, model)

    text_emb = text_emb.unsqueeze(0)  # (1, d)
    device = next(model.parameters()).device

    img_emb_list = []
    valid_names = []

    for name in images:
        valid_names.append(name)
        if name not in image_dict:
            d = model.config.projection_dim
            img_emb_list.append(torch.zeros(d, device=device))
            continue

        img = image_dict[name]
        if use_augmentations:
            emb = get_image_embedding_multi_view(img, processor=processor, model=model, temp=temp)
        else:
            emb = get_image_embedding_single_view(img, processor=processor, model=model, temp=temp)
        img_emb_list.append(emb)

    img_feats = torch.stack(img_emb_list, dim=0)
    sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
    ranked_indices = np.argsort(sims)[::-1]
    ranked_images = [valid_names[i] for i in ranked_indices]
    return ranked_images


def evaluate_subset_configurable(
    df: pd.DataFrame,
    image_dict: Dict[str, Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
    *,
    use_prompting: bool,
    use_augmentations: bool,
    photo_weight: float = 0.6,
    semantic_weight: float = 0.4,
    temp: float = 0.7,
) -> Tuple[float, float]:
    ranks = []
    for _, row in df.iterrows():
        target = row["target"]
        sentence = row["sentence"]
        images = [row[f"image_{i}"] for i in range(10)]
        label = row["label"]

        ranked_images = choose_image_configurable(
            target=target,
            sentence=sentence,
            images=images,
            image_dict=image_dict,
            model=model,
            processor=processor,
            use_prompting=use_prompting,
            use_augmentations=use_augmentations,
            photo_weight=photo_weight,
            semantic_weight=semantic_weight,
            temp=temp,
        )
        rank = ranked_images.index(label) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    hit = float(np.mean(ranks == 1))
    return mrr, hit


def setup_clip_and_nltk(model_name="openai/clip-vit-base-patch32"):
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return device, model, processor

def profile_inference_cost_for_config(
    df,
    image_dict,
    model,
    processor,
    *,
    use_prompting: bool,
    use_augmentations: bool,
    photo_weight=0.6,
    semantic_weight=0.4,
    temp=0.7,
    num_rows=50,
    warmup=5,
    seed=42,
    out_size=(224, 224),
):
    """
    Profiles computational cost for the current ablation config using core.py functions.
    - If use_prompting=True  => dual-channel prompting text embedding
    - else                  => plain text embedding
    - If use_augmentations=True => multi-view image embedding
    - else                     => single-view image embedding
    Measures:
      1) augmented views per image (for the augmentations pipeline)
      2) text embedding latency
      3) image embedding latency
      4) end-to-end latency for one ranking query (10 candidates)
    """

    import time
    import numpy as np
    import torch

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

    # count views *only* if augmentations are enabled
    def _count_views(img):
        views = []
        views.extend(generate_tta_views(img, num_random_augs=2, out_size=out_size))
        views.extend(multi_crops(img, out_size=out_size))
        views.extend(grid_patches(img, grid_size=3, out_size=out_size))
        views.extend(center_saliency_crops(img, out_size=out_size))
        views.extend(mid_quadrant_crops(img, out_size=out_size))
        return len(views)

    device = _device_of(model)
    model.eval()

    df = df.reset_index(drop=True)
    if len(df) == 0:
        print("[profile] df is empty; nothing to profile.")
        return

    rng = np.random.default_rng(seed)
    n = min(num_rows, len(df))
    sample_idxs = rng.choice(len(df), size=n, replace=False)
    sample_df = df.iloc[sample_idxs]

    # views per image
    if use_augmentations:
        views_per_image = None
        for _, row in sample_df.iterrows():
            candidates = [row[f"image_{i}"] for i in range(10)]
            name = next((c for c in candidates if c in image_dict), None)
            if name is not None:
                views_per_image = _count_views(image_dict[name])
                break
        if views_per_image is None:
            views_per_image = 0
    else:
        views_per_image = 1  # single-view baseline

    # warmup
    for _ in range(max(0, warmup)):
        r = sample_df.iloc[0]
        candidates = [r[f"image_{i}"] for i in range(10)]
        # text
        if use_prompting:
            _ = get_dual_channel_text_embedding(
                target=r["target"], sentence=r["sentence"],
                processor=processor, model=model,
                photo_weight=photo_weight, semantic_weight=semantic_weight
            )
        else:
            _ = get_plain_text_embedding(
                target=r["target"], sentence=r["sentence"],
                processor=processor, model=model
            )
        # one image embed
        for c in candidates:
            if c in image_dict:
                img = image_dict[c]
                if use_augmentations:
                    _ = get_image_embedding_multi_view(img, processor=processor, model=model, temp=temp)
                else:
                    _ = get_image_embedding_single_view(img, processor=processor, model=model, temp=temp)
                break
        # end-to-end ranking
        _ = choose_image_configurable(
            target=r["target"],
            sentence=r["sentence"],
            images=candidates,
            image_dict=image_dict,
            model=model,
            processor=processor,
            use_prompting=use_prompting,
            use_augmentations=use_augmentations,
            photo_weight=photo_weight,
            semantic_weight=semantic_weight,
            temp=temp,
        )
    _sync(device)

    # timing lists
    text_times = []
    image_times = []
    query_times = []

    for _, row in sample_df.iterrows():
        target = row["target"]
        sentence = row["sentence"]
        candidates = [row[f"image_{i}"] for i in range(10)]

        # text timing
        _sync(device)
        t0 = time.perf_counter()
        if use_prompting:
            _ = get_dual_channel_text_embedding(
                target=target, sentence=sentence,
                processor=processor, model=model,
                photo_weight=photo_weight, semantic_weight=semantic_weight
            )
        else:
            _ = get_plain_text_embedding(
                target=target, sentence=sentence,
                processor=processor, model=model
            )
        _sync(device)
        t1 = time.perf_counter()
        text_times.append(t1 - t0)

        # image timing (each candidate)
        for name in candidates:
            img = image_dict.get(name, None)
            if img is None:
                continue
            _sync(device)
            i0 = time.perf_counter()
            if use_augmentations:
                _ = get_image_embedding_multi_view(img, processor=processor, model=model, temp=temp)
            else:
                _ = get_image_embedding_single_view(img, processor=processor, model=model, temp=temp)
            _sync(device)
            i1 = time.perf_counter()
            image_times.append(i1 - i0)

        # end-to-end timing
        _sync(device)
        q0 = time.perf_counter()
        _ = choose_image_configurable(
            target=target,
            sentence=sentence,
            images=candidates,
            image_dict=image_dict,
            model=model,
            processor=processor,
            use_prompting=use_prompting,
            use_augmentations=use_augmentations,
            photo_weight=photo_weight,
            semantic_weight=semantic_weight,
            temp=temp,
        )
        _sync(device)
        q1 = time.perf_counter()
        query_times.append(q1 - q0)

    ts = _stats(text_times)
    ims = _stats(image_times)
    qs = _stats(query_times)

    est_img_cost_per_query = ims["mean"] * 10.0 if not np.isnan(ims["mean"]) else float("nan")

    print("\n==============================")
    print("COMPUTATIONAL COST PROFILE")
    print("==============================")
    print(f"Config: prompting={use_prompting} | augmentations={use_augmentations}")
    print(f"Device: {device}")
    print(f"Profiled queries: {n}")
    print(f"Candidates/query: 10")
    print(f"Views per image: {views_per_image}")
    print("")
    print("Text embedding latency:")
    print(f"  Mean: {_ms(ts['mean']):.2f} ms | P50: {_ms(ts['p50']):.2f} | P90: {_ms(ts['p90']):.2f} | P95: {_ms(ts['p95']):.2f}")
    print("")
    print("Image embedding latency:")
    print(f"  Mean: {_ms(ims['mean']):.2f} ms/image | P50: {_ms(ims['p50']):.2f} | P90: {_ms(ims['p90']):.2f} | P95: {_ms(ims['p95']):.2f}")
    print(f"  Est. image-embedding per query (10 candidates): {_ms(est_img_cost_per_query):.2f} ms/query")
    print("")
    print("End-to-end ranking latency (10 candidates):")
    print(f"  Mean: {_ms(qs['mean']):.2f} ms/query | P50: {_ms(qs['p50']):.2f} | P90: {_ms(qs['p90']):.2f} | P95: {_ms(qs['p95']):.2f}")
    print("==============================\n")
