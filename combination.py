# Ambiguous Image Selection with CLIP + WordNet + Prompting + Image Augmentations

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt

import spacy
import nltk
from nltk.corpus import wordnet as wn
from transformers import CLIPModel, CLIPProcessor


############################## Utilities ##############################


def spacy_to_wordnet_pos(spacy_pos: str):
    """Map spaCy POS tags to WordNet POS tags."""
    if spacy_pos in ["NOUN", "PROPN"]:
        return wn.NOUN
    elif spacy_pos == "VERB":
        return wn.VERB
    elif spacy_pos == "ADJ":
        return wn.ADJ
    elif spacy_pos == "ADV":
        return wn.ADV
    else:
        return None


############################## Data Loading ##############################


def load_data(file_path, train_val, target_size=(384, 384), use_cache=True):
    """
    Load the SemEval data and images, with optional image caching.

    Args:
        file_path (str): Base path to the dataset directory.
        train_val (str): 'train', 'test', or 'trial'.
        target_size (tuple): Size to initially resize images (for storage).
        use_cache (bool): Whether to use cached images (pickle).

    Returns:
        data (DataFrame): Columns: target, sentence, image_0-9, label
        image_dict (dict): {filename -> PIL.Image (resized)}
    """
    path = os.path.join(file_path, train_val + "_v1")
    cache_file = os.path.join(path, "image_cache.pkl")

    # Load text data
    path_data = os.path.join(path, f"{train_val}.data.v1.txt")
    data = pd.read_csv(path_data, sep="\t", header=None)
    data.columns = ["target", "sentence"] + [
        f"image_{i}" for i in range(data.shape[1] - 2)
    ]

    # Load labels
    path_labels = os.path.join(path, f"{train_val}.gold.v1.txt")
    with open(path_labels, "r") as f:
        gold_labels = [line.strip() for line in f]
    data["label"] = gold_labels

    # Try cached images
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached images from {cache_file}...")
        with open(cache_file, "rb") as f:
            image_dict = pickle.load(f)
        print(f"Loaded {len(image_dict)} cached images")
        return data, image_dict

    # Otherwise, load images from disk
    path_images = os.path.join(path, train_val + "_images_v1")
    image_dict = {}
    files = os.listdir(path_images)
    for filename in tqdm(
        files, total=len(files), desc="Loading images", unit="image"
    ):
        if filename.lower().endswith((".jpg", ".png")):
            try:
                img = Image.open(os.path.join(path_images, filename)).convert("RGB")
                img_resized = img.resize(target_size, resample=Image.BICUBIC)
                image_dict[filename] = img_resized
            except Exception:
                continue

    # Save cache
    if use_cache:
        print(f"Saving images to cache: {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(image_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cached {len(image_dict)} images")

    return data, image_dict


############################## Text Embeddings ##############################


def get_sentence_embedding(text, tokenizer=None, model=None):
    """
    Get a CLIP text embedding (preferred) or a fallback transformer embedding.

    Supports:
        - text: str
        - text: list[str] (multiple prompts → averaged embedding)
    """
    if isinstance(tokenizer, CLIPProcessor) and isinstance(model, CLIPModel):
        model.eval()
        device = next(model.parameters()).device

        # Normalize to list of strings
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        with torch.no_grad():
            inputs = tokenizer(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_text_features(**inputs)  # (N, d)
            feats = F.normalize(feats, p=2, dim=-1)
            emb = feats.mean(dim=0)  # average prompts
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    # Fallback path if not using CLIP (not used in your current pipeline)
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        last_hidden = outputs.last_hidden_state  # (1, T, d)
        embedding = last_hidden.mean(dim=1)[0]
        embedding = F.normalize(embedding, p=2, dim=-1)
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
    hflip = img.transpose(Image.FLIP_LEFT_RIGHT).resize(
        out_size, resample=Image.BICUBIC
    )
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


def grid_patches(
    img: Image.Image, grid_size=3, out_size=(224, 224)
) -> list[Image.Image]:
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
    temp: float = 0.7,
    out_size=(224, 224),
) -> torch.Tensor:
    """
    Build a rich multi-view embedding:
    - Strong TTA views
    - Multi-crops (center + corners + zoomed)
    - Grid patches
    - Center-focused crops
    Then:
    - Batch all views through CLIP
    - Average features (mean)
    - Normalize + temperature scale + re-normalize
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
        use_amp = (
            isinstance(device, torch.device) and device.type == "cuda"
        )
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
        feat_mean = F.normalize(feat_mean, p=2, dim=-1)

        # Temperature scaling sharpens cosine similarities
        feat_scaled = feat_mean / temp
        feat_final = F.normalize(feat_scaled, p=2, dim=-1)

    return feat_final


############################## WordNet + NER WSD ##############################


def choose_definition(
    target,
    sentence,
    tokenizer=None,
    model=None,
    print_output=False,
    ner=None,
    filter_for_pos=True,
):
    """
    Given a target word and its context sentence, choose the WordNet definition
    whose CLIP text embedding is most similar to the sentence's CLIP embedding.

    Returns:
        best_syn: Best WordNet synset (or None)
        best_emb: Embedding for best definition (or context embedding if no synsets)
        context_embedding: Embedding for the full sentence
    """

    if ner is None:
        ner = spacy.load("en_core_web_sm")

    if tokenizer is None or model is None:
        raise ValueError("Tokenizer and model (e.g. CLIPProcessor, CLIPModel) must be provided.")

    # Sanitize inputs
    if target is None or (isinstance(target, float) and np.isnan(target)):
        target = ""
    if not isinstance(target, str):
        target = str(target)

    if sentence is None or (isinstance(sentence, float) and np.isnan(sentence)):
        sentence = ""
    if not isinstance(sentence, str):
        sentence = str(sentence)

    # Context embedding: the full sentence
    context_embedding = get_sentence_embedding(sentence, tokenizer=tokenizer, model=model)
    context_embedding = F.normalize(context_embedding, p=2, dim=-1)  # (d,)

    synsets = wn.synsets(target)
    if not synsets:
        if print_output:
            print(f"No synsets found for '{target}', falling back to sentence embedding.")
        return None, context_embedding, context_embedding

    # spaCy POS → WordNet POS
    doc = ner(sentence)
    pos = None
    for tok in doc:
        if tok.text.lower() == target.lower():
            pos = tok.pos_
            break
    wordnet_pos = spacy_to_wordnet_pos(pos) if pos is not None else None

    if filter_for_pos and wordnet_pos is not None:
        filtered_synsets = [syn for syn in synsets if syn.pos() == wordnet_pos]
        if len(filtered_synsets) == 0:
            filtered_synsets = synsets
    else:
        filtered_synsets = synsets

    definition_embeddings = []
    for syn in filtered_synsets:
        definition = syn.definition()
        definition_embedding = get_sentence_embedding(
            definition, tokenizer=tokenizer, model=model
        )
        definition_embedding = F.normalize(definition_embedding, p=2, dim=-1)
        definition_embeddings.append((syn, definition_embedding))

    def_matrix = torch.stack([emb for _, emb in definition_embeddings], dim=0)
    sims = torch.matmul(def_matrix, context_embedding)  # (N,)

    best_idx = torch.argmax(sims).item()
    best_syn, best_emb = definition_embeddings[best_idx]

    if print_output:
        print("The cosine similarities for each definition are:")
        for i, syn in enumerate(filtered_synsets):
            print(f"{sims[i].item():.4f} {syn.name()} : {syn.definition()}")
        print("Best sense:", best_syn.name())
        print("Definition:", best_syn.definition())

    return best_syn, best_emb, context_embedding


def build_final_text_embedding(
    target,
    sentence,
    best_syn,
    best_definition_emb,
    context_emb,
    processor,
    model,
    ner=None,
    embedding_weights=(0.7, 0.2, 0.1),  # definition, context, prompts
):
    """
    Build the final text embedding using:
    - WordNet best-sense definition embedding
    - Context (sentence) embedding
    - Prompt-based embedding (keywords + explicit task instructions)
    """

    if ner is None:
        ner = spacy.load("en_core_web_sm")

    # Extract keywords from sentence (NOUN, VERB, ADJ)
    doc = ner(sentence)
    keywords = " ".join(
        [tok.text for tok in doc if tok.pos_ in ["NOUN", "VERB", "ADJ"]]
    )
    if not keywords.strip():
        keywords = target

    # Prompt set for CLIP (short, visual, not LLM-style)
    final_prompts = [
        f"a photo illustrating the meaning of '{target}' as '{best_syn.definition()}' in '{sentence}'",
        f"a realistic image showing '{target}' in the sense of '{best_syn.definition()}'",
        f"an image that matches the usage of '{target}' in: {sentence}",
        f"important visual concepts: {keywords}",
    ]

    prompt_emb = get_sentence_embedding(final_prompts, tokenizer=processor, model=model)

    # Weighted fusion
    def_w, ctx_w, prm_w = embedding_weights
    final = def_w * best_definition_emb + ctx_w * context_emb + prm_w * prompt_emb
    final = F.normalize(final, p=2, dim=-1)
    return final.unsqueeze(0)  # (1, d)


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
    embedding_weights_for_fusion=(0.85, 0.15),
    use_prompt_fusion=True,
    print_output=False,
):
    """
    choose_image():
    - WordNet + spaCy POS to pick best synset
    - Sentence context embedding
    - Optional prompt fusion using definition + keywords
    - Rich multi-view CLIP image embeddings
    """

    if ner is None:
        ner = spacy.load("en_core_web_sm")

    if not (isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel)):
        raise ValueError("Processor and model must be CLIPProcessor and CLIPModel.")

    # 1. WSD + embeddings for definition + context
    if print_output:
        print("\nSentence:", sentence)
        print("Target:", target)

    best_syn, best_definition_emb, context_emb = choose_definition(
        target,
        sentence,
        tokenizer=processor,
        model=model,
        print_output=print_output,
        ner=ner,
        filter_for_pos=filter_for_pos,
    )

    # 2. Build text embedding
    if best_definition_emb is None or best_syn is None:
        # Fallback to context only
        text_emb = context_emb.unsqueeze(0)
    else:
        if use_prompt_fusion:
            # Use prompting-aware fusion
            text_emb = build_final_text_embedding(
                target,
                sentence,
                best_syn,
                best_definition_emb,
                context_emb,
                processor,
                model,
                ner=ner,
                embedding_weights=(0.7, 0.2, 0.1),
            )
        else:
            # Simple weighted fusion: definition + context
            def_w, ctx_w = embedding_weights_for_fusion
            blended = def_w * best_definition_emb + ctx_w * context_emb
            blended = F.normalize(blended, p=2, dim=-1)
            text_emb = blended.unsqueeze(0)

    # 3. Compute image embeddings (multi-view CLIP)
    img_emb_list = []
    valid_names = []

    for name in images:
        valid_names.append(name)
        if name not in image_dict:
            d = model.config.projection_dim
            img_emb_list.append(torch.zeros(d, device=next(model.parameters()).device))
            continue

        img = image_dict[name]
        emb = get_image_embedding(
            img,
            processor=processor,
            model=model,
            temp=0.7,
        )
        img_emb_list.append(emb)

    img_feats = torch.stack(img_emb_list, dim=0)  # (N, d)

    # 4. Rank images by cosine similarity
    sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
    ranked_indices = np.argsort(sims)[::-1]
    ranked_images = [valid_names[i] for i in ranked_indices]

    if print_output:
        for rank, i in enumerate(ranked_indices):
            name = valid_names[i]
            if name in image_dict:
                plt.imshow(image_dict[name])
                title = f"Rank {rank+1} | sim={sims[i]:.4f}\nSentence: {sentence}"
                if best_syn is not None:
                    title += f"\nSense: {best_syn.name()} | Def: {best_syn.definition()}"
                plt.title(title)
                plt.axis("off")
                plt.show()
        print("Ranked Images:", ranked_images)

    ranked_captions = [None for _ in ranked_images]
    ranked_embs = [None for _ in ranked_images]
    return ranked_images, ranked_captions, ranked_embs


############################## Main ##############################


if __name__ == "__main__":

    # Device selection
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    file_path = "dataset"
    print_output = False

    # Download WordNet data if needed
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    ner = spacy.load("en_core_web_sm")

    # For CLIP: 224x224 is natural, but cached images can be 384x384; we re-resize later anyway
    data, image_dict = load_data(
        file_path=file_path,
        train_val="trial",
        target_size=(384, 384),
        use_cache=True,
    )

    predicted_ranks = []
    for idx, row in data.iterrows():
        target = row["target"]
        sentence = row["sentence"]
        images = [row[f"image_{i}"] for i in range(10)]
        label = row["label"]

        ranked_images, ranked_captions, ranked_embs = choose_image(
            target,
            sentence,
            images,
            image_dict,
            tokenizer=None,
            model=model,
            processor=processor,
            blip_model=None,
            ner=ner,
            filter_for_pos=False,
            embedding_weights_for_fusion=(0.85, 0.15),
            use_prompt_fusion=True,
            print_output=print_output,
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
