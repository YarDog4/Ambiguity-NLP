# Dual-channel text ensemble + rich image augmentation + quadrant crops

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import random
import nltk
import pickle
from nltk.corpus import wordnet as wn
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F


############################## Load in the SemEval data ##############################

def load_data(file_path, train_val="trial", target_size=(384, 384), use_cache=True):
    """Load in the data

    Args:
        file_path (str): The file path
        train_val (str): Whether to load in the train, test, or trial set
        target_size (tuple): The size of each image to use
            Use (224, 224) for CLIP only, (384, 384) for BLIP
        use_cache (bool): Whether to use cached images if available

    Returns:
        data (DataFrame): Target, Sentence, image_0-9, label
        image_dict (dict): Map image name to image
    """
    # Train/trial/test set
    path = os.path.join(file_path, train_val+"_v1")
    
    # Cache file path
    cache_file = os.path.join(path, f"image_cache.pkl")

    # Load in the data
    path_data = os.path.join(path, train_val+".data.v1.txt")
    data = pd.read_csv(path_data, sep='\t', header=None)
    data.columns = ['target', 'sentence'] + [f'image_{i}' for i in range(data.shape[1] - 2)]

    # Load in the labels
    path_labels = os.path.join(path, train_val+".gold.v1.txt")
    with open(path_labels, "r") as f: 
        gold_labels = [line.strip() for line in f]
    data['label'] = gold_labels

    # Try to load cached images
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached images from {cache_file}...")
        with open(cache_file, 'rb') as f:
            image_dict = pickle.load(f)
        print(f"Loaded {len(image_dict)} cached images")
        return data, image_dict

    # Load in the images (first time or if cache disabled)
    path_images = os.path.join(path, train_val+"_images_v1")
    image_dict = {}
    files = os.listdir(path_images)
    for filename in tqdm(files, total=len(files), desc="Loading in the Images", unit="image"):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'): 
            try:
                img = Image.open(os.path.join(path_images, filename)).convert('RGB')
                img_resized = img.resize(target_size, resample=Image.BICUBIC)
                image_dict[filename] = img_resized
            except Exception:
                continue

    # Save to cache
    if use_cache:
        print(f"Saving images to cache: {cache_file}...")
        with open(cache_file, 'wb') as f:
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
    'Photo-style' prompts – what CLIP saw a lot of during training.
    """
    target = str(target).strip()
    sentence = "" if sentence is None else str(sentence).strip()

    context = extract_context_word(target, sentence).strip()
    base_context = context if context else sentence

    prompts: list[str] = []

    # prompts.append(f"a photo of {target} {base_context}".strip())
    # prompts.append(f"{target} {base_context}, realistic photo".strip())
    # prompts.append(f"{target} near {base_context}, real world".strip())
    # prompts.append(f"{target} with {base_context}, natural scene".strip())
    # prompts.append(f"{target} appearing in a {base_context} environment".strip())
    # prompts.append(f"{target} {base_context}, high quality photograph".strip())
    prompts.append(f"A photo of {target}")
    prompts.append(f"An image of {target}")
    if base_context:
        prompts.append(f"A photo of {target} in the context of {base_context}")
        prompts.append(f"{target} related to {base_context}, real world")

    # synonym-boosted variant
    # t_syn = get_synonym(target)
    # c_syn = get_synonym(context) if context else None
    # syn_target = t_syn if t_syn else target
    # syn_context = c_syn if c_syn else base_context
    # if syn_target and syn_context:
    #     prompts.append(f"a photo of {syn_target} {syn_context}".strip())

    t_syn = get_synonym(target)
    if t_syn:
        prompts.append(f"A photo of {t_syn}")
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

    # prompts.append(f"{target} related to {base_context}".strip())
    # prompts.append(f"the concept of {target} in {base_context}".strip())
    # prompts.append(f"{target} in the context of {base_context}".strip())
    # prompts.append(f"visual sense of {target} in {base_context}".strip())
    # prompts.append(f"illustration of {target} used with {base_context}".strip())
    prompts.append(f"{target}, visual concept")
    if base_context:
        prompts.append(f"{target} in the context of {base_context}")
        prompts.append(f"The idea of {target} related to {base_context}")


    # sentence-level rewrites
    # if sentence:
    #     prompts.append(f"a visual depiction of: {sentence}".strip())
    #     prompts.append(f"illustration of the phrase: {sentence}".strip())
    #     prompts.append(f"{sentence} (image)".strip())
    #     prompts.append(f"{sentence}, realistic photography".strip())
    if sentence:
        prompts.append(f"A visual depiction of: {sentence}")

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

def add_small_noise(x: torch.Tensor, scale: float = 0.01) -> torch.Tensor:
    """
    Tiny Gaussian noise to reduce brittleness of prompt representations.
    """
    if scale <= 0:
        return x
    noise = scale * torch.randn_like(x)
    return x + noise

def get_dual_channel_text_embedding(
    target: str,
    sentence: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    photo_weight: float = 0.6,
    semantic_weight: float = 0.4,
    noise_scale: float = 0.01,
) -> torch.Tensor:
    """
    Dual-channel text embedding:
      - photo-style prompts channel
      - semantic/context prompts channel
    Then weighted combination + normalization.
    """
    photo_prompts = build_photo_prompts(target, sentence)
    sem_prompts = build_semantic_prompts(target, sentence)

    photo_emb = get_prompted_text_embedding(photo_prompts, processor, model)
    sem_emb = get_prompted_text_embedding(sem_prompts, processor, model)

    combined = photo_weight * photo_emb + semantic_weight * sem_emb
    combined = F.normalize(combined, p=2, dim=-1)

    combined = add_small_noise(combined, scale=noise_scale)
    combined = F.normalize(combined, p=2, dim=-1) #double normalization

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


# def generate_tta_views(
#     img: Image.Image,
#     num_random_augs: int = 3,
#     out_size=(224, 224),
# ) -> list[Image.Image]:
#     """
#     Strong TTA for CLIP:
#     - original
#     - horizontal flip
#     - center crop
#     - zoomed center
#     - grayscale
#     - a few random geometric+photometric augmentations
#     """
#     views = []
#     w, h = img.size

#     # Base resized view
#     base = img.resize(out_size, resample=Image.BICUBIC)
#     views.append(base)

#     # Horizontal flip
#     hflip = img.transpose(Image.FLIP_LEFT_RIGHT).resize(out_size, resample=Image.BICUBIC)
#     views.append(hflip)

#     # Center crop (80%)
#     crop_scale = 0.8
#     cw, ch = int(w * crop_scale), int(h * crop_scale)
#     left = (w - cw) // 2
#     top = (h - ch) // 2
#     center_crop = img.crop((left, top, left + cw, top + ch))
#     center_crop = center_crop.resize(out_size, resample=Image.BICUBIC)
#     views.append(center_crop)

#     # Zoomed center (60%)
#     zoom_scale = 0.6
#     zw, zh = int(w * zoom_scale), int(h * zoom_scale)
#     zleft = (w - zw) // 2
#     ztop = (h - zh) // 2
#     zoom_crop = img.crop((zleft, ztop, zleft + zw, ztop + zh))
#     zoom_crop = zoom_crop.resize(out_size, resample=Image.BICUBIC)
#     views.append(zoom_crop)

#     # Grayscale variant
#     gray = img.convert("L").convert("RGB").resize(out_size, resample=Image.BICUBIC)
#     views.append(gray)

#     # A few random augmentations (geometric + photometric)
#     for _ in range(num_random_augs):
#         aug = random_geometric_augment(img)
#         aug = random_photometric_augment(aug)
#         aug = aug.resize(out_size, resample=Image.BICUBIC)
#         views.append(aug)

#     return views

def generate_tta_views(
    img: Image.Image,
    out_size=(224, 224),
) -> list[Image.Image]:
    """
    Lightweight TTA (5 views):
      - original resized
      - horizontal flip
      - center crop
      - slight zoomed center
      - grayscale
    This is intentionally small to avoid overfitting + keep speed.
    """
    views: list[Image.Image] = []
    w, h = img.size

    # 1) Base
    base = img.resize(out_size, resample=Image.BICUBIC)
    views.append(base)

    # 2) Horizontal flip
    hflip = img.transpose(Image.FLIP_LEFT_RIGHT).resize(out_size, resample=Image.BICUBIC)
    views.append(hflip)

    # 3) Center crop (80%)
    crop_scale = 0.8
    cw, ch = int(w * crop_scale), int(h * crop_scale)
    left = (w - cw) // 2
    top = (h - ch) // 2
    center_crop = img.crop((left, top, left + cw, top + ch))
    center_crop = center_crop.resize(out_size, resample=Image.BICUBIC)
    views.append(center_crop)

    # 4) Zoomed center (60%)
    zoom_scale = 0.6
    zw, zh = int(w * zoom_scale), int(h * zoom_scale)
    zleft = (w - zw) // 2
    ztop = (h - zh) // 2
    zoom_crop = img.crop((zleft, ztop, zleft + zw, ztop + zh))
    zoom_crop = zoom_crop.resize(out_size, resample=Image.BICUBIC)
    views.append(zoom_crop)

    # 5) Grayscale
    gray = img.convert("L").convert("RGB").resize(out_size, resample=Image.BICUBIC)
    views.append(gray)

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
    # temp: float = 0.7,
    out_size=(224, 224),
    use_multi_crop=True,
    use_saliency=True,
    use_quadrants=True,
    use_grid_patches=True,
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
    - Normalize + temperature scale
    """
    model.eval()
    device = next(model.parameters()).device

    # Collect all views
    views: list[Image.Image] = []
    views.extend(generate_tta_views(img, out_size=out_size))
    if use_multi_crop:
        views.extend(multi_crops(img, out_size=out_size))
    if use_grid_patches:
        views.extend(grid_patches(img, grid_size=3, out_size=out_size))
    if use_saliency:
        views.extend(center_saliency_crops(img, out_size=out_size))
    if use_quadrants:
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
        # feat_scaled = feat_mean / temp
        feat_final = F.normalize(feat_mean, p=2, dim=-1)

    return feat_final


############################## Choosing Images ##############################

def choose_image(
    target,
    sentence,
    images,
    image_dict,
    tokenizer=None,
    model=None,
    processor=None,
    blip_model=None,
    ner=None,
    filter_for_pos=True,
    embedding_weights=None,
    print_output=False,
    noise_scale=0.00,
    use_saliency=True,
    use_quadrants=True,
    use_multi_crop = True,
    use_grid_patches = True
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
        photo_weight=0.5,
        semantic_weight=0.5,
        noise_scale = 0.01
    )
    text_emb = text_emb.unsqueeze(0)  # (1, d)
    
    text_emb = text_emb + noise_scale * torch.randn_like(text_emb)
    text_emb = F.normalize(text_emb, p=2, dim=-1)

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
            out_size=(224,224),
            use_multi_crop=use_multi_crop,
            use_saliency=use_saliency,
            use_quadrants=use_quadrants,
            use_grid_patches=use_grid_patches,
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


############################## Main ##############################

# if __name__ == "__main__":

#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     file_path = "dataset"
#     print_output = False

#     # Download WordNet data if needed (for synonyms)
#     nltk.download("wordnet", quiet=True)
#     nltk.download("omw-1.4", quiet=True)

#     model_name = "openai/clip-vit-base-patch32"
#     processor = CLIPProcessor.from_pretrained(model_name)
#     model = CLIPModel.from_pretrained(model_name).to(device)

#     data, image_dict = load_data(file_path=file_path, train_val="trial")

#     predicted_ranks = []
#     for idx, row in data.iterrows():
#         target = row["target"]
#         sentence = row["sentence"]
#         images = [row[f"image_{i}"] for i in range(10)]
#         label = row["label"]

#         ranked_images, ranked_captions, ranked_embs = choose_image(
#             target,
#             sentence,
#             images,
#             image_dict,
#             tokenizer=None,
#             model=model,
#             processor=processor,
#             blip_model=None,
#             ner=None,
#             filter_for_pos=False,
#             embedding_weights=None,
#             print_output=print_output,
#         )

#         predicted_rank = ranked_images.index(label) + 1
#         print("Predicted Rank:", predicted_rank)
#         predicted_ranks.append(predicted_rank)

#     predicted_ranks = np.array(predicted_ranks)
#     mrr = np.mean(1 / predicted_ranks)
#     hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)

#     print("---------------------------------")
#     print(f"MRR: {mrr}")
#     print(f"Hit Rate: {hit_rate}")

# if __name__ == "__main__":

#     # Device setup
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     file_path = "dataset"
#     print_output = False

#     nltk.download("wordnet", quiet=True)
#     nltk.download("omw-1.4", quiet=True)

#     # Load ViT-L/14
#     model_name = "openai/clip-vit-base-patch32"
#     processor = CLIPProcessor.from_pretrained(model_name)
#     model = CLIPModel.from_pretrained(model_name).to(device)

#     # Load TRIAL data
#     data, image_dict = load_data(file_path=file_path, train_val="trial")

#     # -------------------------------------------
#     # INTERNAL VALIDATION SPLIT (80/20)
#     # -------------------------------------------
#     np.random.seed(42)
#     indices = np.arange(len(data))
#     np.random.shuffle(indices)

#     split = int(len(indices) * 0.8)
#     train_idx = indices[:split]
#     val_idx = indices[split:]

#     train_data = data.iloc[train_idx].reset_index(drop=True)
#     val_data = data.iloc[val_idx].reset_index(drop=True)

#     print(f"[INFO] Training size: {len(train_data)}  | Validation size: {len(val_data)}")

#     # -------------------------------------------
#     # HYPERPARAMETER SEARCH SPACE
#     # -------------------------------------------
#     search_space = [
#         (0.6, 0.4, 0.01),
#         (0.5, 0.5, 0.01),
#         (0.7, 0.3, 0.01),
#         (0.6, 0.4, 0.03),
#         (0.5, 0.5, 0.03),
#     ]

#     def evaluate_subset(df, photo_w, semantic_w, noise_scale):
#         """Compute MRR + Hit Rate on a chosen subset."""
#         ranks = []
#         for _, row in df.iterrows():
#             target = row["target"]
#             sentence = row["sentence"]
#             images = [row[f"image_{i}"] for i in range(10)]
#             label = row["label"]

#             ranked_images, _, _ = choose_image(
#                 target=target,
#                 sentence=sentence,
#                 images=images,
#                 image_dict=image_dict,
#                 model=model,
#                 processor=processor,
#             )

#             ranks.append(ranked_images.index(label) + 1)

#         ranks = np.array(ranks)
#         mrr = np.mean(1.0 / ranks)
#         hit = np.mean(ranks == 1)
#         return mrr, hit

#     # -------------------------------------------
#     # SEARCH FOR BEST HP ON INTERNAL VALIDATION
#     # -------------------------------------------
#     best_score = -1
#     best_hparams = None

#     print("\n[INFO] Starting hyperparameter search...\n")

#     for (pw, sw, ns) in search_space:
#         print(f"Testing: photo={pw}, semantic={sw}, noise={ns}")

#         # Monkey-patch hyperparameters inside choose_image call
#         global_photo_weight = pw
#         global_semantic_weight = sw
#         global_noise_scale = ns

#         mrr, hit = evaluate_subset(val_data, pw, sw, ns)
#         score = mrr + hit  # simple combined objective

#         print(f"  -> MRR={mrr:.4f} | Hit={hit:.4f}")

#         if score > best_score:
#             best_score = score
#             best_hparams = (pw, sw, ns)

#     print("\n[INFO] Best Hyperparameters (Validation)")
#     print("----------------------------------------")
#     print(f"photo_weight={best_hparams[0]}")
#     print(f"semantic_weight={best_hparams[1]}")
#     print(f"noise_scale={best_hparams[2]}")

#     # -------------------------------------------
#     # FINAL EVAL ON FULL TRIAL SET
#     # -------------------------------------------
#     print("\n[INFO] Running final evaluation on FULL trial set...\n")

#     final_ranks = []
#     for _, row in data.iterrows():
#         target = row["target"]
#         sentence = row["sentence"]
#         images = [row[f"image_{i}"] for i in range(10)]
#         label = row["label"]

#         ranked_images, _, _ = choose_image(
#             target=target,
#             sentence=sentence,
#             images=images,
#             image_dict=image_dict,
#             model=model,
#             processor=processor,
#         )

#         final_ranks.append(ranked_images.index(label) + 1)

#     final_ranks = np.array(final_ranks)
#     final_mrr = np.mean(1.0 / final_ranks)
#     final_hit = np.mean(final_ranks == 1)

#     print("---------------------------------")
#     print(f"FINAL MRR (Trial): {final_mrr}")
#     print(f"FINAL Hit Rate (Trial): {final_hit}")


# --------------------------------------------------------------
# Optuna Objective Function
# --------------------------------------------------------------
def objective(trial, data, image_dict, model, processor):
    """
    Runs one hyperparameter trial.
    Returns the MRR on internal validation split.
    """

    # ------------------------------
    # Sample hyperparameters
    # ------------------------------
    photo_weight   = trial.suggest_float("photo_weight",   0.1, 0.9)
    semantic_weight = trial.suggest_float("semantic_weight", 0.1, 0.9)

    # ensure they sum to ~1
    total = photo_weight + semantic_weight
    photo_weight /= total
    semantic_weight /= total

    noise_scale = trial.suggest_float("noise_scale", 0.0, 0.08)

    # extra knobs (optional)
    use_saliency = trial.suggest_categorical("use_saliency", [True, False])
    use_quadrants = trial.suggest_categorical("use_quadrants", [True, False])
    use_grid_patches = trial.suggest_categorical("use_grid_patches", [True, False])
    use_multi_crop = trial.suggest_categorical("use_multi_crop", [True, False])
    


    # ------------------------------
    # Internal validation split
    # ------------------------------
    val_idx = np.random.choice(len(data), size=int(0.3 * len(data)), replace=False)
    val_data = data.iloc[val_idx]

    predicted_ranks = []

    for _, row in val_data.iterrows():
        target = row["target"]
        sentence = row["sentence"]
        images = [row[f"image_{i}"] for i in range(10)]
        label = row["label"]

        ranked_images, _, _ = choose_image(
            target,
            sentence,
            images,
            image_dict,
            tokenizer=None,
            model=model,
            processor=processor,
            blip_model=None,
            ner=None,
            filter_for_pos=False,
            embedding_weights=[photo_weight, semantic_weight],
            print_output=False,
            noise_scale=noise_scale,
            use_saliency=use_saliency,
            use_quadrants=use_quadrants,
            use_multi_crop=use_multi_crop,
            use_grid_patches=use_grid_patches,
        )

        # rank position
        predicted_rank = ranked_images.index(label) + 1
        predicted_ranks.append(predicted_rank)

    predicted_ranks = np.array(predicted_ranks)

    mrr = np.mean(1 / predicted_ranks)

    # return NEGATIVE error (Optuna minimizes)
    return -mrr

if __name__ == "__main__":

    ###########################################################################
    # Device Setup
    ###########################################################################
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    ###########################################################################
    # Load Dataset
    ###########################################################################
    file_path = "dataset"
    print_output = False

    # WordNet for synonyms (if needed)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    ###########################################################################
    # Load CLIP ViT-L/14 (or your chosen model)
    ###########################################################################
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    print("Loading dataset...")
    data, image_dict = load_data(file_path=file_path, train_val="trial")

    ###########################################################################
    # Bayesian Optimization (Optuna)
    ###########################################################################
    print("\n==============================")
    print("🔥 Starting Bayesian tuning...")
    print("==============================")

    import optuna
    from optuna.samplers import TPESampler

    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",     # because our objective returns -MRR
        sampler=sampler
    )

    # Run optimization
    N_TRIALS = 40   # you can set 100+ for better results

    study.optimize(
        lambda trial: objective(
            trial,
            data=data,
            image_dict=image_dict,
            model=model,
            processor=processor
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    ###########################################################################
    # Print / Save Best Hyperparameters
    ###########################################################################
    print("\n==============================")
    print("🏆 BEST BAYESIAN TRIAL")
    print("==============================")
    best_params = study.best_trial.params
    print(best_params)

    # Optionally save to a file
    with open("best_hyperparams.pkl", "wb") as f:
        pickle.dump(best_params, f)

    ###########################################################################
    # Final Evaluation on Full Trial Set (using best hyperparameters)
    ###########################################################################
    print("\nEvaluating full TRIAL set using best hyperparameters...")

    predicted_ranks = []
    photo_weight = best_params["photo_weight"]
    semantic_weight = best_params["semantic_weight"]
    noise_scale = best_params["noise_scale"]
    use_saliency = best_params["use_saliency"]
    use_quadrants = best_params["use_quadrants"]
    use_multi_crop = best_params["use_multi_crop"]
    use_grid_patches = best_params["use_grid_patches"]


    for idx, row in data.iterrows():
        target = row["target"]
        sentence = row["sentence"]
        images = [row[f"image_{i}"] for i in range(10)]
        label = row["label"]

        ranked_images, ranked_captions, ranked_embs = choose_image(
            target=target,
            sentence=sentence,
            images=images,
            image_dict=image_dict,
            tokenizer=None,
            model=model,
            processor=processor,
            blip_model=None,
            ner=None,
            filter_for_pos=False,
            embedding_weights=[photo_weight, semantic_weight],
            print_output=False,
            noise_scale=noise_scale,
            use_saliency=use_saliency,
            use_quadrants=use_quadrants,
            use_grid_patches=use_grid_patches,
            use_multi_crop=use_multi_crop
        )

        predicted_rank = ranked_images.index(label) + 1
        predicted_ranks.append(predicted_rank)

    predicted_ranks = np.array(predicted_ranks)
    mrr = np.mean(1 / predicted_ranks)
    hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)

    ###########################################################################
    # Print Evaluation Metrics
    ###########################################################################
    print("\n==============================")
    print("📊 FINAL RESULTS (Trial Set)")
    print("==============================")
    print(f"MRR:      {mrr:.6f}")
    print(f"Hit Rate: {hit_rate:.6f}")

#Trial set:
#MRR: 0.8770833333333333
#Hit Rate: 0.8125

#tweaks iteration 1:
# removed grid patches
# softmax-weighted fusion of images
# added two precision boost prompts
# dual-channel style ensemble
# similarity sharpening
# double normalization
#MRR: 0.8302083333333333
#Hit Rate: 0.75
