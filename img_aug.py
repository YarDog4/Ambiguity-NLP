#Image Augmentations
import os
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F


############################## Load in the SemEval data ##############################

def load_data(file_path, train_val="test", target_size=(384, 384)):
    """Load the SemEval dataset."""
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
        if filename.lower().endswith(('.jpg', '.png')):
            try:
                img = Image.open(os.path.join(path_images, filename)).convert('RGB')
                image_dict[filename] = img
            except Exception:
                continue

    return data, image_dict


############################## Text Embeddings ##############################

def get_sentence_embedding(text, tokenizer=None, model=None):
    """Get a CLIP text embedding or a fallback transformer embedding."""
    if isinstance(tokenizer, CLIPProcessor) and isinstance(model, CLIPModel):
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = tokenizer(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_text_features(**inputs)
            features = F.normalize(features, p=2, dim=-1)
        return features[0]

    # Fallback for non-CLIP models (unused in your current pipeline)
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
        last_hidden = outputs.last_hidden_state
        embedding = last_hidden.mean(dim=1)[0]
    return embedding


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


def generate_tta_views(img: Image.Image,
                       num_random_augs: int = 3,
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


def get_image_embedding(img: Image.Image,
                        processor: CLIPProcessor,
                        model: CLIPModel,
                        temp: float = 0.7,
                        out_size=(224, 224)) -> torch.Tensor:
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
    views.extend(generate_tta_views(img, num_random_augs=3, out_size=out_size))
    views.extend(multi_crops(img, out_size=out_size))
    views.extend(grid_patches(img, grid_size=3, out_size=out_size))
    views.extend(center_saliency_crops(img, out_size=out_size))

    with torch.no_grad():
        inputs = processor(images=views, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # If on GPU, use autocast for speed
        use_amp = (device.type == "cuda") if isinstance(device, torch.device) else False
        if hasattr(torch.cuda, "amp"):
            ctx = torch.cuda.amp.autocast(enabled=use_amp)
        else:
            # Fallback if amp not available
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

def choose_image(target, sentence, images, image_dict,
                 tokenizer=None, model=None, processor=None, blip_model=None,
                 print_output=False):
    """Choose the best matching image using CLIP with rich image augmentations."""
    if isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel):

        text_emb = get_sentence_embedding(sentence, tokenizer=processor, model=model)
        text_emb = text_emb.unsqueeze(0)  # (1, d)
        device = next(model.parameters()).device

        img_emb_list = []
        valid_names = []

        for name in images:
            valid_names.append(name)

            if name not in image_dict:
                # Missing image -> zero vector
                d = model.config.projection_dim
                img_emb_list.append(torch.zeros(d, device=device))
                continue

            img = image_dict[name]
            emb = get_image_embedding(img, processor=processor, model=model, temp=0.7)
            img_emb_list.append(emb)

        img_feats = torch.stack(img_emb_list, dim=0)  # (N, d)

        sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
        ranked_indices = np.argsort(sims)[::-1]
        ranked_images = [valid_names[i] for i in ranked_indices]

        if print_output:
            for rank, i in enumerate(ranked_indices):
                if valid_names[i] in image_dict:
                    plt.imshow(image_dict[valid_names[i]])
                    plt.title(f"Rank {rank+1} | sim={sims[i]:.4f}\nSentence: {sentence}")
                    plt.axis('off')
                    plt.show()
            print("Ranked Images:", ranked_images)

        ranked_captions = [None] * len(ranked_images)
        ranked_embs = [None] * len(ranked_images)

        return ranked_images, ranked_captions, ranked_embs


############################## Main ##############################

if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    file_path = "dataset"
    print_output = False

    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    data, image_dict = load_data(file_path=file_path, train_val="trial")

    predicted_ranks = []
    for idx, row in data.iterrows():
        target = row['target']
        sentence = row['sentence']
        images = [row[f'image_{i}'] for i in range(10)]
        label = row['label']

        ranked_images, ranked_captions, ranked_embs = choose_image(
            target, sentence, images, image_dict,
            tokenizer=None, model=model, processor=tokenizer,
            blip_model=None, print_output=print_output
        )

        predicted_rank = ranked_images.index(label) + 1
        print("Predicted Rank:", predicted_rank)
        predicted_ranks.append(predicted_rank)

    predicted_ranks = np.array(predicted_ranks)
    mrr = np.mean(1 / predicted_ranks)
    hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)

    print("---------------------------------")
    print(f"MRR: {mrr}")
    print(f"Hit Rate: {hit_rate}")
