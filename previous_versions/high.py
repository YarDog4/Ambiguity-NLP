import os
import random
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
import spacy
from nltk.corpus import wordnet as wn


##############################
# Load SemEval Data
##############################

def load_data(file_path, train_val="test", target_size=(384, 384)):
    """
    Load the SemEval dataset.

    Expected structure:
        <file_path>/
            train_v1/
                train.data.v1.txt
                train.gold.v1.txt
                train_images_v1/
            trial_v1/
            test_v1/
    """
    path = os.path.join(file_path, train_val + "_v1")

    # Load text data
    path_data = os.path.join(path, train_val + ".data.v1.txt")
    data = pd.read_csv(path_data, sep='\t', header=None)
    data.columns = ['target', 'sentence'] + [
        f'image_{i}' for i in range(data.shape[1] - 2)
    ]

    # Load labels
    path_labels = os.path.join(path, train_val + ".gold.v1.txt")
    with open(path_labels, "r") as f:
        gold_labels = [line.strip() for line in f]
    data['label'] = gold_labels

    # Load images
    path_images = os.path.join(path, train_val + "_images_v1")
    image_dict = {}
    files = os.listdir(path_images)

    for filename in tqdm(files, total=len(files), desc="Loading Images", unit="image"):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                img = Image.open(os.path.join(path_images, filename)).convert('RGB')
                image_dict[filename] = img
            except Exception:
                continue

    return data, image_dict


##############################
# Basic CLIP Text Embeddings
##############################

def get_clip_text_emb(text, processor, model):
    """
    Single text → CLIP embedding (normalized).
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feat = model.get_text_features(**inputs)
        feat = F.normalize(feat, p=2, dim=-1)
    return feat[0]   # (d,)


def get_clip_text_emb_from_list(text_list, processor, model):
    """
    Multiple texts → average CLIP embedding (normalized).
    """
    device = next(model.parameters()).device
    embs = []

    with torch.no_grad():
        for text in text_list:
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feat = model.get_text_features(**inputs)
            feat = F.normalize(feat, p=2, dim=-1)
            embs.append(feat[0])

    embs = torch.stack(embs, dim=0)        # (N, d)
    mean_emb = embs.mean(dim=0)            # (d,)
    mean_emb = F.normalize(mean_emb, p=2, dim=-1)
    return mean_emb


##############################
# Context Word Extraction (spaCy)
##############################

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

    # Case 1: exactly two tokens, one of them is the target
    if len(tokens) == 2:
        t_low = t.lower()
        tok0, tok1 = tokens[0], tokens[1]
        if tok0.lower() == t_low:
            return tok1
        if tok1.lower() == t_low:
            return tok0

    # Otherwise, just treat the whole sentence as "context"
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

def build_micro_prompts(target: str, sentence: str) -> list[str]:
    """
    Build a set of short, CLIP-friendly micro-prompts
    from the target word and its context phrase.
    """
    target = str(target).strip()
    sentence = "" if sentence is None else str(sentence).strip()

    # Core context word/phrase (e.g. 'erosion' from 'bank erosion')
    context = extract_context_word(target, sentence).strip()
    base_context = context if context else sentence

    prompts: list[str] = []

    # 1–11: core micro-prompts
    prompts.append(f"{target} {base_context}".strip())
    prompts.append(f"{base_context} {target}".strip())
    prompts.append(f"{target} {base_context} photo".strip())
    prompts.append(f"{target} {base_context} scene".strip())
    prompts.append(f"{target} near {base_context}".strip())
    prompts.append(f"{target} with {base_context}".strip())
    prompts.append(f"a photo of {target} in {base_context}".strip())
    prompts.append(f"real {target} with {base_context}".strip())
    prompts.append(f"{target} related to {base_context}".strip())
    prompts.append(f"{target} appearing in a {base_context} setting".strip())
    prompts.append(f"visual sense of {target} in {base_context}".strip())

    # 12: synonym-expanded variant
    target_syn = get_synonym(target)
    context_syn = get_synonym(context) if context else None

    syn_target = target_syn if target_syn else target
    syn_context = context_syn if context_syn else base_context

    if syn_target and syn_context:
        prompts.append(f"{syn_target} {syn_context}".strip())

    # Remove any accidental empties
    prompts = [p for p in prompts if p]

    return prompts

def get_text_embedding_from_micro_prompts(
    target: str,
    sentence: str,
    processor: CLIPProcessor,
    model: CLIPModel,
) -> torch.Tensor:
    """
    Encode 12+ micro-prompts with CLIP text encoder,
    average them, and L2-normalize to get a strong text embedding.
    """
    model.eval()
    device = next(model.parameters()).device

    prompts = build_micro_prompts(target, sentence)

    with torch.no_grad():
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)   # (P, d)
        feats = F.normalize(feats, p=2, dim=-1)
        mean_feat = feats.mean(dim=0)              # (d,)
        mean_feat = F.normalize(mean_feat, p=2, dim=-1)

    return mean_feat   # shape (d,)

##############################
# Image Augmentations (your strong pipeline)
##############################

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


def generate_tta_views(img: Image.Image,
                       num_random_augs: int = 2,
                       out_size=(224, 224)) -> list[Image.Image]:
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


def get_image_embedding(
    img: Image.Image,
    processor: CLIPProcessor,
    model: CLIPModel,
    temp: float = 0.65,
    out_size=(224, 224)
) -> torch.Tensor:
    """
    Build a rich multi-view embedding:
    - Strong TTA views
    - Multi-crops (center + corners + zoomed)
    - Grid patches
    - Center-focused crops

    Then:
    - Batch all views through CLIP
    - Average features
    - Normalize + temperature scale
    """
    model.eval()
    device = next(model.parameters()).device

    # Collect all views
    views = []
    views.extend(generate_tta_views(img, num_random_augs=2, out_size=out_size))
    views.extend(multi_crops(img, out_size=out_size))
    views.extend(grid_patches(img, grid_size=3, out_size=out_size))
    views.extend(center_saliency_crops(img, out_size=out_size))

    with torch.no_grad():
        inputs = processor(images=views, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        use_amp = (isinstance(device, torch.device) and device.type == "cuda")
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


##############################
# Main Matching Function
##############################

def choose_image(
    target,
    sentence,
    images,
    image_dict,
    model,
    processor,
    nlp,
    text_weights=(0.60, 0.25, 0.15),
    img_temp=0.65,
    print_output=False
):
    """
    - Build fused text embedding:
        sentence + (target, context_word) + micro-prompts
    - Build strong image embeddings via multi-view CLIP
    - Rank images by cosine similarity
    """
    device = next(model.parameters()).device

    # 1. Build final text embedding
    text_emb = get_text_embedding_from_micro_prompts(
        target=target,
        sentence=sentence,
        processor=processor,
        model=model,
        # nlp=nlp,
        # weights=text_weights,
    )  # (1, d)

    # 2. Compute image embeddings
    img_embs = []
    valid_names = []

    for name in images:
        valid_names.append(name)

        if name not in image_dict:
            d = model.config.projection_dim
            img_embs.append(torch.zeros(d, device=device))
            continue

        img = image_dict[name]
        emb = get_image_embedding(
            img,
            processor=processor,
            model=model,
            temp=img_temp
        )
        img_embs.append(emb)

    img_feats = torch.stack(img_embs, dim=0)  # (N, d)

    # 3. Rank
    sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
    ranked_idx = np.argsort(sims)[::-1]
    ranked_imgs = [valid_names[i] for i in ranked_idx]

    if print_output:
        print(f"\nSentence: {sentence}")
        print(f"Target: {target}")
        print("Ranked images (best → worst):", ranked_imgs)

    ranked_captions = [None] * len(ranked_imgs)
    ranked_embs = [None] * len(ranked_imgs)

    return ranked_imgs, ranked_captions, ranked_embs


##############################
# Main Script
##############################

if __name__ == "__main__":

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    file_path = "dataset"
    print_output = False

    # CLIP model
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    # spaCy (for context word extraction)
    nlp = spacy.load("en_core_web_sm")

    # Load data
    data, image_dict = load_data(file_path=file_path, train_val="trial")

    predicted_ranks = []
    for idx, row in data.iterrows():
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
            nlp=nlp,
            text_weights=(0.60, 0.25, 0.15),  # can tune
            img_temp=0.65,                     # can tune
            print_output=print_output
        )

        predicted_rank = ranked_images.index(label) + 1
        print("Predicted Rank:", predicted_rank)
        predicted_ranks.append(predicted_rank)

    predicted_ranks = np.array(predicted_ranks)
    mrr = np.mean(1.0 / predicted_ranks)
    hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)

    print("---------------------------------")
    print(f"MRR: {mrr}")
    print(f"Hit Rate: {hit_rate}")

#Trial set
#MRR: 0.8458333333333333
#Hit Rate: 0.75