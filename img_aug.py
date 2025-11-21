#Image Augmentations
import os
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import random
import pickle
import spacy
import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F


############################## Load in the SemEval data ##############################
def spacy_to_wordnet_pos(spacy_pos):
    if spacy_pos in ['NOUN', 'PROPN']:
        return wn.NOUN
    elif spacy_pos == 'VERB':
        return wn.VERB
    elif spacy_pos == 'ADJ':
        return wn.ADJ
    elif spacy_pos == 'ADV':
        return wn.ADV
    else:
        return None
    
def load_data(file_path, train_val, target_size=(384, 384), use_cache=True):
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


############################## Text Embeddings ##############################

# def get_sentence_embedding(text, tokenizer=None, model=None):
#     """Get a CLIP text embedding or a fallback transformer embedding."""
#     if isinstance(tokenizer, CLIPProcessor) and isinstance(model, CLIPModel):
#         model.eval()
#         device = next(model.parameters()).device
#         with torch.no_grad():
#             inputs = tokenizer(text=[text], return_tensors="pt", padding=True, truncation=True)
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             features = model.get_text_features(**inputs)
#             features = F.normalize(features, p=2, dim=-1)
#         return features[0]

#     # Fallback for non-CLIP models (unused in your current pipeline)
#     tokens = tokenizer(text, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**tokens)
#         last_hidden = outputs.last_hidden_state
#         embedding = last_hidden.mean(dim=1)[0]
#     return embedding

def get_prompted_text_embeddings(text_list, processor, model):
    """
    Generate a CLIP text embedding for each prompt in text_list,
    then average + normalize.
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

    embs = torch.stack(embs)                 # (num_prompts, d)
    mean_emb = embs.mean(dim=0)              # (d,)
    mean_emb = F.normalize(mean_emb, p=2, dim=-1)
    return mean_emb


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

# def choose_definition(target, sentence, tokenizer=None, model=None, print_output=False, ner=None, filter_for_pos=True):
#     """
#     Given a target word and its context sentence, choose the WordNet definition
#     whose CLIP text embedding is most similar to the sentence's CLIP embedding.

#     Args:
#         target (str): The target word
#         sentence (str): The full sentence containing the target
#         tokenizer: The tokenizer (e.g. CLIPProcessor)
#         model: The text model (e.g. CLIPModel)
#         print_output (bool): Whether to print similarities and chosen sense
#         ner (spacy model): Named Entity Recognition model
#         filter_for_pos (bool): Wether or not to filter for the POS

#     Returns:
#         best_syn: The best WordNet synset (or None if none exist)
#         best_emb (Tensor): The CLIP text embedding of the best definition
#                            (or the sentence embedding if no synsets found)
#         context_embedding (Tensor): The CLIP text embedding of the whole sentence
#     """

#     # Load the pretrained English model
#     if ner is None: ner = spacy.load('en_core_web_sm')

#     if tokenizer is None or model is None:
#         raise ValueError(
#             "Tokenizer and model (e.g. CLIPProcessor and CLIPModel) must be provided."
#         )

#     # Sanitize inputs: ensure strings and handle NaN/None
#     if target is None or (isinstance(target, float) and np.isnan(target)):
#         # No usable target; fall back to sentence embedding
#         target = ""
#     if not isinstance(target, str):
#         target = str(target)
#     if sentence is None or (isinstance(sentence, float) and np.isnan(sentence)):
#         sentence = ""
#     if not isinstance(sentence, str):
#         sentence = str(sentence)

#     # Use the full sentence as context representation
#     context_text = sentence
#     context_embedding = get_sentence_embedding(
#         context_text, tokenizer=tokenizer, model=model
#     )
#     context_embedding = F.normalize(context_embedding, p=2, dim=-1)  # (d,)

#     synsets = wn.synsets(target)
#     if not synsets:
#         # No WordNet entries, fall back to the sentence embedding
#         if print_output:
#             print(
#                 f"No synsets found for '{target}', falling back to sentence embedding."
#             )
#         return None, context_embedding, context_embedding

#     definition_embeddings = []
#     # Process the sentence
#     doc = ner(sentence)
#     for token in doc:  # Find target token and get its POS tag
#         if token.text == target:
#             print(f"Target word: {token.text}, POS tag: {token.pos_}")
#             pos = token.pos_
#             break
#     wordnet_pos = spacy_to_wordnet_pos(pos)

#     if filter_for_pos: filtered_synsets = [syn for syn in synsets if syn.pos() == wordnet_pos]
#     else: filtered_synsets = synsets
#     if len(filtered_synsets)==0: filtered_synsets = synsets
#     for syn in filtered_synsets:
#         definition = syn.definition()
#         definition_embedding = get_sentence_embedding(
#             definition, tokenizer=tokenizer, model=model
#         )
#         definition_embedding = F.normalize(definition_embedding, p=2, dim=-1)  # (d,)
#         definition_embeddings.append((syn, definition_embedding))

#     # Stack definition embeddings and compute cosine similarity with context
#     def_matrix = torch.stack([emb for _, emb in definition_embeddings], dim=0)  # (N, d)
#     # embeddings are normalized -> dot product = cosine similarity
#     sims = torch.matmul(def_matrix, context_embedding)  # (N,)

#     best_idx = torch.argmax(sims).item()
#     best_syn, best_emb = definition_embeddings[best_idx]

#     if print_output:
#         print("The cosine similarities for each definition are:")
#         for i, syn in enumerate(filtered_synsets):
#             print(f"{sims[i].item():.4f} {syn.name()} : {syn.definition()}")
#         print("Best sense:", best_syn.name())
#         print("Definition:", best_syn.definition())

#     return best_syn, best_emb, context_embedding

def choose_definition_prompted(
    target,
    sentence,
    tokenizer,
    model,
    ner=None,
    filter_for_pos=True,
    print_output=False
):
    """
    Uses:
    - WordNet synsets
    - spaCy POS filtering
    - PROMPTED CLIP embeddings for definitions
    - Multiple prompt templates
    """

    # Load spaCy if needed
    if ner is None:
        ner = spacy.load("en_core_web_sm")

    # Clean inputs
    if target is None or str(target).strip() == "":
        return None, None, get_prompted_text_embeddings(sentence, tokenizer, model)

    if not isinstance(sentence, str):
        sentence = str(sentence)

    # Basic context embedding (will be prompted later)
    context_emb = get_prompted_text_embeddings(sentence, tokenizer, model)
    context_emb = F.normalize(context_emb, p=2, dim=-1)

    # Get synsets
    synsets = wn.synsets(target)
    if not synsets:
        return None, context_emb, context_emb

    # POS filtering
    doc = ner(sentence)
    pos = None
    for tok in doc:
        if tok.text.lower() == target.lower():
            pos = tok.pos_
            break

    wn_pos = spacy_to_wordnet_pos(pos)
    if filter_for_pos and wn_pos is not None:
        filtered = [s for s in synsets if s.pos() == wn_pos]
        if len(filtered) == 0:
            filtered = synsets
    else:
        filtered = synsets

    # PROMPTED definition embeddings
    def_embs = []
    for syn in filtered:
        definition = syn.definition()

        # Multi-prompt ensemble
        prompts = [
            f"In this sentence: '{sentence}', the word '{target}' means: {definition}.",
            f"The meaning of '{target}' in this context is: {definition}.",
            f"A photo showing the meaning '{definition}' of '{target}' in: {sentence}.",
            f"Depict visually the sense '{definition}' for the word '{target}'.",
        ]

        emb = get_prompted_text_embeddings(prompts, tokenizer, model)
        def_embs.append((syn, emb))

    # Compare to context
    mat = torch.stack([emb for (_, emb) in def_embs], dim=0)
    sims = torch.matmul(mat, context_emb)

    best_idx = sims.argmax().item()
    best_syn, best_emb = def_embs[best_idx]

    return best_syn, best_emb, context_emb

def build_final_text_embedding(
    target,
    sentence,
    best_syn,
    best_definition_emb,
    context_emb,
    processor,
    model,
    ner=None,
    embedding_weights=[0.4, 0.4, 0.2],   # syn, context, keywords
):
    """
    Builds the FINAL text embedding using:
    - Synset definition prompt
    - Context prompt
    - Keyword prompt (important nouns/verbs)
    - Weighted fusion
    """

    if ner is None:
        ner = spacy.load("en_core_web_sm")

    # Extract keywords
    doc = ner(sentence)
    keywords = " ".join([tok.text for tok in doc if tok.pos_ in ["NOUN", "VERB", "ADJ"]])
    if keywords.strip() == "":
        keywords = target

    # Build multiple final prompts
    final_prompts = [
        f"Select an image that shows the meaning of '{target}' as '{best_syn.definition()}' in this sentence: '{sentence}'.",
        f"A photo representing the sense '{best_syn.definition()}' of the word '{target}'.",
        f"An image illustrating the usage of '{target}' meaning '{best_syn.definition()}' in context.",
        f"Important visual concepts: {keywords}. Choose an image matching this meaning.",
    ]

    final_prompt_emb = get_prompted_text_embeddings(final_prompts, processor, model)

    # Weighted fusion
    syn_w, ctx_w, kw_w = embedding_weights

    final = (
        syn_w * best_definition_emb +
        ctx_w * context_emb +
        kw_w * final_prompt_emb
    )

    final = F.normalize(final, p=2, dim=-1)
    return final.unsqueeze(0)    # (1, d)


############################## Choosing Images ##############################

# def choose_image(target, sentence, images, image_dict, tokenizer=None, model=None, processor=None,
#     blip_model=None, ner=None, filter_for_pos=True, embedding_weights=[0.5, 0.5], print_output=False):
#     """
#     Combined choose_image():
#     - Uses WordNet + POS + spacy NER for choosing correct sense of target word
#     - Uses sense definition + sentence embedding fusion
#     - Uses rich multi-view CLIP image embeddings (generate_tta_views, multi_crops, grid_patches...)
#     """

#     # Load spaCy if not provided
#     if ner is None:
#         import spacy
#         ner = spacy.load("en_core_web_sm")

#     # Ensure correct pipeline: CLIP only
#     if not (isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel)):
#         raise ValueError("Processor and model must be CLIPProcessor and CLIPModel.")

#     device = next(model.parameters()).device

#     # 1. WSD STEP — choose best definition using WordNet + CLIP
#     if print_output:
#         print("\nSentence:", sentence)
#         print("Target:", target)

#     best_syn, best_definition_emb, context_emb = choose_definition_prompted(
#         target,
#         sentence,
#         tokenizer=processor,     # CLIPProcessor
#         model=model,             # CLIPModel
#         print_output=print_output,
#         ner=ner,
#         filter_for_pos=filter_for_pos
#     )

#     # If WordNet found no valid senses, fall back to sentence-embedding only
#     if best_definition_emb is None:
#         text_emb = context_emb.unsqueeze(0)
#     else:
#         # Weighted fusion of definition embedding + context embedding
#         blended = embedding_weights[0] * best_definition_emb + embedding_weights[1] * context_emb
#         text_emb = blended.unsqueeze(0)   # shape (1, d)

#     # 2. Compute image embeddings — MULTI-VIEW CLIP VERSION
#     img_emb_list = []
#     valid_names = []

#     for name in images:
#         valid_names.append(name)

#         if name not in image_dict:
#             # missing image fallback
#             d = model.config.projection_dim
#             img_emb_list.append(torch.zeros(d, device=device))
#             continue

#         img = image_dict[name]

#         emb = get_image_embedding(
#             img,
#             processor=processor,
#             model=model,
#             temp=0.7
#         )
#         img_emb_list.append(emb)

#     # Stack into (N, d)
#     img_feats = torch.stack(img_emb_list, dim=0)

#     # 3. Rank images by cosine similarity
#     sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
#     ranked_indices = np.argsort(sims)[::-1]
#     ranked_images = [valid_names[i] for i in ranked_indices]

#     # Optional visualization
#     if print_output:
#         for rank, i in enumerate(ranked_indices):
#             if valid_names[i] in image_dict:
#                 plt.imshow(image_dict[valid_names[i]])
#                 title = f"Rank {rank+1} | sim={sims[i]:.4f}\nSentence: {sentence}"
#                 if best_syn is not None:
#                     title += f"\nSense: {best_syn.name()} | Def: {best_syn.definition()}"
#                 plt.title(title)
#                 plt.axis("off")
#                 plt.show()

#         print("Ranked Images:", ranked_images)

#     ranked_captions = [None for _ in ranked_images]
#     ranked_embs = [None for _ in ranked_images]

#     return ranked_images, ranked_captions, ranked_embs

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
    embedding_weights=[0.4, 0.4, 0.2],
    print_output=False
):
    if ner is None:
        ner = spacy.load("en_core_web_sm")

    device = next(model.parameters()).device

    # 1. Prompted WSD
    best_syn, best_def_emb, context_emb = choose_definition_prompted(
        target,
        sentence,
        tokenizer=processor,
        model=model,
        print_output=print_output,
        ner=ner,
        filter_for_pos=filter_for_pos
    )

    # 2. If WSD fails → fallback to context prompt only
    if best_syn is None:
        text_emb = context_emb.unsqueeze(0)
    else:
        text_emb = build_final_text_embedding(
            target,
            sentence,
            best_syn,
            best_def_emb,
            context_emb,
            processor,
            model,
            ner=ner,
            embedding_weights=embedding_weights
        )

    # 3. Compute image embeddings
    img_embs = []
    valid_names = []

    for name in images:
        valid_names.append(name)

        if name not in image_dict:
            d = model.config.projection_dim
            img_embs.append(torch.zeros(d, device=device))
            continue

        img = image_dict[name]
        emb = get_image_embedding(img, processor=processor, model=model, temp=0.7)
        img_embs.append(emb)

    img_feats = torch.stack(img_embs, dim=0)

    # 4. Ranking
    sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
    ranked_idx = np.argsort(sims)[::-1]
    ranked_imgs = [valid_names[i] for i in ranked_idx]

    if print_output:
        print("Ranked:", ranked_imgs)

    return ranked_imgs, [None]*len(ranked_imgs), [None]*len(ranked_imgs)


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

    # Download WordNet data if needed
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    ner = spacy.load('en_core_web_sm')


    data, image_dict = load_data(file_path=file_path, train_val="test")

    predicted_ranks = []
    for idx, row in data.iterrows():
        target = row['target']
        sentence = row['sentence']
        images = [row[f'image_{i}'] for i in range(10)]
        label = row['label']

        ranked_images, ranked_captions, ranked_embs = choose_image(
            target, sentence, images, image_dict,
            tokenizer=None, model=model, processor=processor,
            blip_model=None, ner=ner, filter_for_pos=False, 
            embedding_weights=[0.4, 0.4 ,0.2], print_output=print_output, 
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

    
# ============================================
# Imports
# ============================================
# import os
# import torch
# import pandas as pd
# import numpy as np
# from PIL import Image, ImageFilter, ImageEnhance
# import matplotlib.pyplot as plt
# import random
# import spacy
# import nltk
# from nltk.corpus import wordnet as wn

# from tqdm import tqdm
# from transformers import CLIPModel, CLIPProcessor
# import torch.nn.functional as F


# # ============================================
# # Utilities: POS mapping & data loading
# # ============================================
# def spacy_to_wordnet_pos(spacy_pos):
#     """Map spaCy POS tag to WordNet POS tag."""
#     if spacy_pos in ["NOUN", "PROPN"]:
#         return wn.NOUN
#     elif spacy_pos == "VERB":
#         return wn.VERB
#     elif spacy_pos == "ADJ":
#         return wn.ADJ
#     elif spacy_pos == "ADV":
#         return wn.ADV
#     else:
#         return None


# def load_data(file_path, train_val="test", target_size=(384, 384)):
#     """Load the SemEval dataset."""
#     path = os.path.join(file_path, train_val + "_v1")

#     # Load text data
#     path_data = os.path.join(path, train_val + ".data.v1.txt")
#     data = pd.read_csv(path_data, sep="\t", header=None)
#     data.columns = ["target", "sentence"] + [
#         f"image_{i}" for i in range(data.shape[1] - 2)
#     ]

#     # Load labels
#     path_labels = os.path.join(path, train_val + ".gold.v1.txt")
#     with open(path_labels, "r") as f:
#         gold_labels = [line.strip() for line in f]
#     data["label"] = gold_labels

#     # Load images
#     path_images = os.path.join(path, train_val + "_images_v1")
#     image_dict = {}
#     files = os.listdir(path_images)

#     for filename in tqdm(files, total=len(files), desc="Loading Images", unit="image"):
#         if filename.lower().endswith((".jpg", ".png")):
#             try:
#                 img = Image.open(os.path.join(path_images, filename)).convert("RGB")
#                 image_dict[filename] = img
#             except Exception:
#                 continue

#     return data, image_dict


# # ============================================
# # Text Embedding & Prompting
# # ============================================
# def get_prompted_text_embeddings(text_list, processor, model):
#     """
#     Generate CLIP text embeddings for a list of prompts, then average + normalize.

#     Args:
#         text_list (list[str]): list of prompt strings
#         processor: CLIPProcessor
#         model: CLIPModel

#     Returns:
#         torch.Tensor: (d,) normalized embedding
#     """
#     device = next(model.parameters()).device
#     embs = []

#     with torch.no_grad():
#         for text in text_list:
#             inputs = processor(
#                 text=[text], return_tensors="pt", padding=True, truncation=True
#             )
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             feat = model.get_text_features(**inputs)  # (1, d)
#             feat = F.normalize(feat, p=2, dim=-1)
#             embs.append(feat[0])  # (d,)

#     embs = torch.stack(embs, dim=0)  # (num_prompts, d)
#     mean_emb = embs.mean(dim=0)      # (d,)
#     mean_emb = F.normalize(mean_emb, p=2, dim=-1)
#     return mean_emb


# def get_context_embedding(target, sentence, processor, model):
#     """
#     Build a context embedding using multiple prompts about the sentence + target.
#     """
#     if sentence is None:
#         sentence = ""
#     if target is None:
#         target = ""

#     prompts = [
#         f"In this sentence: '{sentence}', focus on the word '{target}'.",
#         f"Consider the sentence: '{sentence}'. The important word is '{target}'.",
#         f"An image should match the meaning of '{target}' as used in: '{sentence}'.",
#     ]

#     return get_prompted_text_embeddings(prompts, processor, model)


# # ============================================
# # Image Augmentations & Embeddings
# # ============================================
# def random_geometric_augment(img: Image.Image) -> Image.Image:
#     """Mild random geometric transforms: flip, rotation, slight zoom/crop."""
#     w, h = img.size
#     out = img

#     # Random horizontal flip (50% chance)
#     if random.random() < 0.5:
#         out = out.transpose(Image.FLIP_LEFT_RIGHT)

#     # Small random rotation
#     angle = random.uniform(-7, 7)  # degrees
#     out = out.rotate(angle, resample=Image.BICUBIC, expand=False)

#     # Slight random crop (95–100% of size)
#     scale = random.uniform(0.95, 1.0)
#     new_w, new_h = int(w * scale), int(h * scale)
#     if new_w < w and new_h < h:
#         left = random.randint(0, w - new_w)
#         top = random.randint(0, h - new_h)
#         out = out.crop((left, top, left + new_w, top + new_h))
#         out = out.resize((w, h), resample=Image.BICUBIC)

#     return out


# def random_photometric_augment(img: Image.Image) -> Image.Image:
#     """Mild brightness/contrast/color/blur jitter."""
#     out = img

#     # Brightness
#     if random.random() < 0.5:
#         enhancer = ImageEnhance.Brightness(out)
#         factor = random.uniform(0.95, 1.05)
#         out = enhancer.enhance(factor)

#     # Contrast
#     if random.random() < 0.5:
#         enhancer = ImageEnhance.Contrast(out)
#         factor = random.uniform(0.95, 1.05)
#         out = enhancer.enhance(factor)

#     # Color
#     if random.random() < 0.5:
#         enhancer = ImageEnhance.Color(out)
#         factor = random.uniform(0.95, 1.05)
#         out = enhancer.enhance(factor)

#     # Slight blur (very mild)
#     if random.random() < 0.3:
#         out = out.filter(ImageFilter.GaussianBlur(radius=0.3))

#     return out


# def generate_tta_views(
#     img: Image.Image, num_random_augs: int = 3, out_size=(224, 224)
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
#     hflip = img.transpose(Image.FLIP_LEFT_RIGHT).resize(
#         out_size, resample=Image.BICUBIC
#     )
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

#     # Random augmentations
#     for _ in range(num_random_augs):
#         aug = random_geometric_augment(img)
#         aug = random_photometric_augment(aug)
#         aug = aug.resize(out_size, resample=Image.BICUBIC)
#         views.append(aug)

#     return views


# def multi_crops(img: Image.Image, out_size=(224, 224)) -> list[Image.Image]:
#     """Center + four corners + zoomed center (multi-crop)."""
#     w, h = img.size
#     crops = []

#     # Main center crop (80%)
#     crop_w, crop_h = int(w * 0.8), int(h * 0.8)
#     crop_w = max(1, crop_w)
#     crop_h = max(1, crop_h)

#     # Center
#     left = (w - crop_w) // 2
#     top = (h - crop_h) // 2
#     center = img.crop((left, top, left + crop_w, top + crop_h))
#     crops.append(center.resize(out_size, resample=Image.BICUBIC))

#     # Top-left
#     tl = img.crop((0, 0, crop_w, crop_h))
#     crops.append(tl.resize(out_size, resample=Image.BICUBIC))

#     # Top-right
#     tr = img.crop((w - crop_w, 0, w, crop_h))
#     crops.append(tr.resize(out_size, resample=Image.BICUBIC))

#     # Bottom-left
#     bl = img.crop((0, h - crop_h, crop_w, h))
#     crops.append(bl.resize(out_size, resample=Image.BICUBIC))

#     # Bottom-right
#     br = img.crop((w - crop_w, h - crop_h, w, h))
#     crops.append(br.resize(out_size, resample=Image.BICUBIC))

#     # Zoomed-in center (60%)
#     zoom_scale = 0.6
#     z_w, z_h = int(w * zoom_scale), int(h * zoom_scale)
#     z_left = (w - z_w) // 2
#     z_top = (h - z_h) // 2
#     zoom_center = img.crop((z_left, z_top, z_left + z_w, z_top + z_h))
#     crops.append(zoom_center.resize(out_size, resample=Image.BICUBIC))

#     return crops


# def grid_patches(
#     img: Image.Image, grid_size=3, out_size=(224, 224)
# ) -> list[Image.Image]:
#     """Split the image into a grid (e.g., 3x3) and return patches."""
#     w, h = img.size
#     patch_w = w // grid_size
#     patch_h = h // grid_size

#     patches = []
#     for gy in range(grid_size):
#         for gx in range(grid_size):
#             left = gx * patch_w
#             top = gy * patch_h
#             right = w if gx == grid_size - 1 else (left + patch_w)
#             bottom = h if gy == grid_size - 1 else (top + patch_h)

#             patch = img.crop((left, top, right, bottom))
#             patches.append(patch.resize(out_size, resample=Image.BICUBIC))

#     return patches


# def center_saliency_crops(img: Image.Image, out_size=(224, 224)) -> list[Image.Image]:
#     """
#     Lightweight 'saliency-like' center crops: smaller crops around the center.
#     Not true saliency, but emphasizes likely subject region.
#     """
#     w, h = img.size
#     crops = []

#     for scale in [0.7, 0.5]:
#         cw, ch = int(w * scale), int(h * scale)
#         left = (w - cw) // 2
#         top = (h - ch) // 2
#         crop = img.crop((left, top, left + cw, top + ch))
#         crops.append(crop.resize(out_size, resample=Image.BICUBIC))

#     return crops


# def get_image_embedding(
#     img: Image.Image,
#     processor: CLIPProcessor,
#     model: CLIPModel,
#     temp: float = 0.7,
#     out_size=(224, 224),
# ) -> torch.Tensor:
#     """
#     Build a rich multi-view embedding:
#     - Strong TTA views
#     - Multi-crops (center + corners + zoomed)
#     - Grid patches
#     - Center-focused crops
#     Then:
#     - Batch all views through CLIP
#     - Average features
#     - Normalize + temperature scale
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     # Collect all views
#     views = []
#     views.extend(generate_tta_views(img, num_random_augs=3, out_size=out_size))
#     views.extend(multi_crops(img, out_size=out_size))
#     views.extend(grid_patches(img, grid_size=3, out_size=out_size))
#     views.extend(center_saliency_crops(img, out_size=out_size))

#     with torch.no_grad():
#         inputs = processor(images=views, return_tensors="pt")
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         # If on GPU, use autocast for speed
#         use_amp = isinstance(device, torch.device) and device.type == "cuda"
#         if hasattr(torch.cuda, "amp"):
#             ctx = torch.cuda.amp.autocast(enabled=use_amp)
#         else:
#             class DummyCtx:
#                 def __enter__(self): return None
#                 def __exit__(self, exc_type, exc_val, exc_tb): return False
#             ctx = DummyCtx()

#         with ctx:
#             feats = model.get_image_features(**inputs)  # (N, d)

#         feats = F.normalize(feats, p=2, dim=-1)
#         feat_mean = feats.mean(dim=0)  # (d,)

#         # Temperature scaling sharpens cosine similarities
#         feat_scaled = feat_mean / temp
#         feat_final = F.normalize(feat_scaled, p=2, dim=-1)

#     return feat_final


# # ============================================
# # WordNet-based WSD with Prompting
# # ============================================
# def choose_definition_prompted(
#     target,
#     sentence,
#     tokenizer,
#     model,
#     ner=None,
#     filter_for_pos=True,
#     print_output=False,
# ):
#     """
#     Uses:
#     - WordNet synsets
#     - spaCy POS filtering
#     - PROMPTED CLIP embeddings for definitions
#     - Multi-prompt ensemble
#     """
#     if ner is None:
#         ner = spacy.load("en_core_web_sm")

#     # Clean inputs
#     if sentence is None:
#         sentence = ""
#     if target is None:
#         target = ""

#     # Context embedding via prompts
#     context_emb = get_context_embedding(target, sentence, tokenizer, model)
#     context_emb = F.normalize(context_emb, p=2, dim=-1)

#     synsets = wn.synsets(target)
#     if not synsets:
#         # No WordNet entries -> just use context embedding
#         if print_output:
#             print(f"No synsets found for '{target}', using context only.")
#         return None, None, context_emb

#     # POS filtering via spaCy
#     doc = ner(sentence)
#     pos = None
#     for tok in doc:
#         if tok.text.lower() == target.lower():
#             pos = tok.pos_
#             break

#     wn_pos = spacy_to_wordnet_pos(pos)
#     if filter_for_pos and wn_pos is not None:
#         filtered = [s for s in synsets if s.pos() == wn_pos]
#         if len(filtered) == 0:
#             filtered = synsets
#     else:
#         filtered = synsets

#     # Prompted definition embeddings
#     def_embs = []
#     for syn in filtered:
#         definition = syn.definition()

#         prompts = [
#             f"In this sentence: '{sentence}', the word '{target}' means: {definition}.",
#             f"The meaning of '{target}' in this context is: {definition}.",
#             f"A photo showing the meaning '{definition}' of '{target}' in: '{sentence}'.",
#             f"Depict visually the sense '{definition}' for the word '{target}'.",
#         ]

#         emb = get_prompted_text_embeddings(prompts, tokenizer, model)
#         def_embs.append((syn, emb))

#     mat = torch.stack([emb for (_, emb) in def_embs], dim=0)  # (N, d)
#     sims = torch.matmul(mat, context_emb)                     # (N,)

#     best_idx = sims.argmax().item()
#     best_syn, best_emb = def_embs[best_idx]

#     if print_output:
#         print(f"Best synset for '{target}': {best_syn.name()} - {best_syn.definition()}")

#     return best_syn, best_emb, context_emb


# def build_final_text_embedding(
#     target,
#     sentence,
#     best_syn,
#     best_definition_emb,
#     context_emb,
#     processor,
#     model,
#     ner=None,
#     embedding_weights=[0.4, 0.4, 0.2],  # syn, context, keyword prompts
# ):
#     """
#     Builds the FINAL text embedding using:
#     - Synset definition prompt
#     - Context prompt
#     - Keyword prompt (important nouns/verbs)
#     - Weighted fusion
#     """
#     if ner is None:
#         ner = spacy.load("en_core_web_sm")

#     if sentence is None:
#         sentence = ""
#     if target is None:
#         target = ""

#     # Extract keywords (nouns/verbs/adjectives)
#     doc = ner(sentence)
#     keywords = " ".join(
#         [tok.text for tok in doc if tok.pos_ in ["NOUN", "VERB", "ADJ"]]
#     )
#     if keywords.strip() == "":
#         keywords = target

#     # Final prompts emphasizing sense & keywords
#     final_prompts = [
#         f"Select an image that shows the meaning of '{target}' as '{best_syn.definition()}' in this sentence: '{sentence}'.",
#         f"A photo representing the sense '{best_syn.definition()}' of the word '{target}'.",
#         f"An image illustrating the usage of '{target}' meaning '{best_syn.definition()}' in context.",
#         f"Important visual concepts: {keywords}. Choose an image matching this meaning.",
#     ]

#     final_prompt_emb = get_prompted_text_embeddings(final_prompts, processor, model)

#     syn_w, ctx_w, kw_w = embedding_weights

#     final = (
#         syn_w * best_definition_emb +
#         ctx_w * context_emb +
#         kw_w * final_prompt_emb
#     )
#     final = F.normalize(final, p=2, dim=-1)

#     return final.unsqueeze(0)  # (1, d)


# # ============================================
# # Image Selection (Main Logic)
# # ============================================
# def choose_image(
#     target,
#     sentence,
#     images,
#     image_dict,
#     tokenizer=None,
#     model=None,
#     processor=None,
#     blip_model=None,   # unused; kept for compatibility
#     ner=None,
#     filter_for_pos=True,
#     embedding_weights=[0.4, 0.4, 0.2],
#     print_output=False,
# ):
#     """
#     Combined choose_image():
#     - WordNet + POS + spaCy NER for sense disambiguation
#     - Prompted CLIP text embeddings
#     - Rich multi-view CLIP image embeddings
#     """
#     if ner is None:
#         ner = spacy.load("en_core_web_sm")

#     if not (isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel)):
#         raise ValueError("processor must be CLIPProcessor and model must be CLIPModel.")

#     device = next(model.parameters()).device

#     # 1. Word sense disambiguation with prompting
#     best_syn, best_def_emb, context_emb = choose_definition_prompted(
#         target,
#         sentence,
#         tokenizer=processor,
#         model=model,
#         ner=ner,
#         filter_for_pos=filter_for_pos,
#         print_output=print_output,
#     )

#     # 2. Build final text embedding
#     if best_syn is None or best_def_emb is None:
#         # No valid synset -> just use context embedding
#         text_emb = context_emb.unsqueeze(0)
#     else:
#         text_emb = build_final_text_embedding(
#             target,
#             sentence,
#             best_syn,
#             best_def_emb,
#             context_emb,
#             processor,
#             model,
#             ner=ner,
#             embedding_weights=embedding_weights,
#         )

#     # 3. Compute image embeddings with augmentations
#     img_embs = []
#     valid_names = []

#     for name in images:
#         valid_names.append(name)

#         if name not in image_dict:
#             d = model.config.projection_dim
#             img_embs.append(torch.zeros(d, device=device))
#             continue

#         img = image_dict[name]
#         emb = get_image_embedding(
#             img,
#             processor=processor,
#             model=model,
#             temp=0.7,
#             out_size=(224, 224),
#         )
#         img_embs.append(emb)

#     img_feats = torch.stack(img_embs, dim=0)  # (N, d)

#     # 4. Rank by cosine similarity
#     sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
#     ranked_idx = np.argsort(sims)[::-1]
#     ranked_imgs = [valid_names[i] for i in ranked_idx]

#     if print_output:
#         print(f"\nSentence: {sentence}")
#         print(f"Target: {target}")
#         print("Ranked images:", ranked_imgs)

#     ranked_captions = [None] * len(ranked_imgs)
#     ranked_embs = [None] * len(ranked_imgs)

#     return ranked_imgs, ranked_captions, ranked_embs


# # ============================================
# # Main Script
# # ============================================
# if __name__ == "__main__":

#     # Device
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     file_path = "dataset"
#     print_output = False

#     # Download WordNet data if needed
#     nltk.download("wordnet", quiet=True)
#     nltk.download("omw-1.4", quiet=True)

#     # CLIP model + processor
#     model_name = "openai/clip-vit-base-patch32"
#     processor = CLIPProcessor.from_pretrained(model_name)
#     model = CLIPModel.from_pretrained(model_name).to(device)

#     # spaCy NER/POS
#     ner = spacy.load("en_core_web_sm")

#     # Load SemEval data
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
#             ner=ner,
#             filter_for_pos=False,              # you can set True to enforce POS match
#             embedding_weights=[0.4, 0.4, 0.2], # syn, context, keyword
#             print_output=print_output,
#         )

#         predicted_rank = ranked_images.index(label) + 1
#         print("Predicted Rank:", predicted_rank)
#         predicted_ranks.append(predicted_rank)

#     predicted_ranks = np.array(predicted_ranks)
#     mrr = np.mean(1.0 / predicted_ranks)
#     hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)

#     print("---------------------------------")
#     print(f"MRR: {mrr}")
#     print(f"Hit Rate: {hit_rate}")
# import os
# import torch
# import pandas as pd
# import numpy as np
# from PIL import Image, ImageFilter, ImageEnhance
# import matplotlib.pyplot as plt
# import random

# from tqdm import tqdm
# from transformers import CLIPModel, CLIPProcessor
# import torch.nn.functional as F


# # ============================================================
# #                 DATA LOADING
# # ============================================================

# def load_data(file_path, train_val="test", target_size=(384, 384)):
#     """
#     Load the SemEval dataset with:
#       - target, sentence
#       - image_0 ... image_9
#       - gold label (correct image filename)
#     """
#     path = os.path.join(file_path, train_val + "_v1")

#     # Load text data
#     path_data = os.path.join(path, train_val + ".data.v1.txt")
#     data = pd.read_csv(path_data, sep="\t", header=None)
#     data.columns = ["target", "sentence"] + [
#         f"image_{i}" for i in range(data.shape[1] - 2)
#     ]

#     # Load labels
#     path_labels = os.path.join(path, train_val + ".gold.v1.txt")
#     with open(path_labels, "r") as f:
#         gold_labels = [line.strip() for line in f]
#     data["label"] = gold_labels

#     # Load images
#     path_images = os.path.join(path, train_val + "_images_v1")
#     image_dict = {}
#     files = os.listdir(path_images)

#     for filename in tqdm(files, total=len(files), desc="Loading Images", unit="image"):
#         if filename.lower().endswith((".jpg", ".png")):
#             try:
#                 img = Image.open(os.path.join(path_images, filename)).convert("RGB")
#                 image_dict[filename] = img
#             except Exception:
#                 continue

#     return data, image_dict


# # ============================================================
# #                 TEXT EMBEDDINGS (PROMPTED)
# # ============================================================

# def get_clip_text_embeddings(text_list, processor, model):
#     """
#     Compute CLIP text embeddings for a list of strings and return
#     a normalized tensor of shape (len(text_list), d).
#     """
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         inputs = processor(text=text_list, return_tensors="pt", padding=True,
#                            truncation=True)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         feats = model.get_text_features(**inputs)  # (N, d)
#         feats = F.normalize(feats, p=2, dim=-1)
#     return feats


# def build_text_embedding(target, sentence, processor, model,
#                          weights=(0.5, 0.3, 0.2)):
#     """
#     Final text embedding built from:
#       1) The raw sentence
#       2) A short visual anchor: "a photo of {target}"
#       3) A context-aware anchor:
#            "a photo of {target} in the context of: {sentence}"

#     weights = (w_sentence, w_anchor, w_context_anchor)
#     """
#     sentence = "" if sentence is None else str(sentence)
#     target = "" if target is None else str(target)

#     base = sentence.strip()
#     if base == "":
#         base = target

#     prompt_anchor = f"a photo of {target}".strip()
#     prompt_context = f"a photo of {target} in the context of: {sentence}".strip()

#     texts = [base, prompt_anchor, prompt_context]
#     feats = get_clip_text_embeddings(texts, processor, model)  # (3, d)

#     w = torch.tensor(weights, device=feats.device, dtype=feats.dtype).unsqueeze(1)
#     fused = (w * feats).sum(dim=0)         # (d,)
#     fused = F.normalize(fused, p=2, dim=-1)
#     return fused  # (d,)


# # ============================================================
# #                 IMAGE AUGMENTATIONS
# # ============================================================

# def random_geometric_augment(img: Image.Image) -> Image.Image:
#     """Mild random geometric transforms: flip, rotation, slight zoom/crop."""
#     w, h = img.size
#     out = img

#     # Random horizontal flip (50% chance)
#     if random.random() < 0.5:
#         out = out.transpose(Image.FLIP_LEFT_RIGHT)

#     # Small random rotation
#     angle = random.uniform(-7, 7)  # degrees
#     out = out.rotate(angle, resample=Image.BICUBIC, expand=False)

#     # Slight random crop (95–100% of size)
#     scale = random.uniform(0.95, 1.0)
#     new_w, new_h = int(w * scale), int(h * scale)
#     if new_w < w and new_h < h:
#         left = random.randint(0, w - new_w)
#         top = random.randint(0, h - new_h)
#         out = out.crop((left, top, left + new_w, top + new_h))
#         out = out.resize((w, h), resample=Image.BICUBIC)

#     return out


# def random_photometric_augment(img: Image.Image) -> Image.Image:
#     """Mild brightness/contrast/color/blur jitter."""
#     out = img

#     # Brightness
#     if random.random() < 0.5:
#         from PIL import ImageEnhance
#         enhancer = ImageEnhance.Brightness(out)
#         factor = random.uniform(0.95, 1.05)
#         out = enhancer.enhance(factor)

#     # Contrast
#     if random.random() < 0.5:
#         from PIL import ImageEnhance
#         enhancer = ImageEnhance.Contrast(out)
#         factor = random.uniform(0.95, 1.05)
#         out = enhancer.enhance(factor)

#     # Color
#     if random.random() < 0.5:
#         from PIL import ImageEnhance
#         enhancer = ImageEnhance.Color(out)
#         factor = random.uniform(0.95, 1.05)
#         out = enhancer.enhance(factor)

#     # Slight blur
#     if random.random() < 0.3:
#         out = out.filter(ImageFilter.GaussianBlur(radius=0.3))

#     return out


# def generate_tta_views(img: Image.Image,
#                        num_random_augs: int = 3,
#                        out_size=(224, 224)) -> list[Image.Image]:
#     """
#     Strong TTA for CLIP:
#       - original
#       - horizontal flip
#       - center crop
#       - zoomed center
#       - grayscale
#       - a few random geometric+photometric augmentations
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

#     # A few random augmentations
#     for _ in range(num_random_augs):
#         aug = random_geometric_augment(img)
#         aug = random_photometric_augment(aug)
#         aug = aug.resize(out_size, resample=Image.BICUBIC)
#         views.append(aug)

#     return views


# def multi_crops(img: Image.Image, out_size=(224, 224)) -> list[Image.Image]:
#     """Center + four corners + zoomed center (multi-crop)."""
#     w, h = img.size
#     crops = []

#     # Main center crop (80%)
#     crop_w, crop_h = int(w * 0.8), int(h * 0.8)
#     crop_w = max(1, crop_w)
#     crop_h = max(1, crop_h)

#     # Center
#     left = (w - crop_w) // 2
#     top = (h - crop_h) // 2
#     center = img.crop((left, top, left + crop_w, top + crop_h))
#     crops.append(center.resize(out_size, resample=Image.BICUBIC))

#     # Top-left
#     tl = img.crop((0, 0, crop_w, crop_h))
#     crops.append(tl.resize(out_size, resample=Image.BICUBIC))

#     # Top-right
#     tr = img.crop((w - crop_w, 0, w, crop_h))
#     crops.append(tr.resize(out_size, resample=Image.BICUBIC))

#     # Bottom-left
#     bl = img.crop((0, h - crop_h, crop_w, h))
#     crops.append(bl.resize(out_size, resample=Image.BICUBIC))

#     # Bottom-right
#     br = img.crop((w - crop_w, h - crop_h, w, h))
#     crops.append(br.resize(out_size, resample=Image.BICUBIC))

#     # Zoomed-in center (60%)
#     zoom_scale = 0.6
#     z_w, z_h = int(w * zoom_scale), int(h * zoom_scale)
#     z_left = (w - z_w) // 2
#     z_top = (h - z_h) // 2
#     zoom_center = img.crop((z_left, z_top, z_left + z_w, z_top + z_h))
#     crops.append(zoom_center.resize(out_size, resample=Image.BICUBIC))

#     return crops


# def grid_patches(img: Image.Image, grid_size=3, out_size=(224, 224)) -> list[Image.Image]:
#     """Split the image into a grid (e.g., 3x3) and return patches."""
#     w, h = img.size
#     patch_w = w // grid_size
#     patch_h = h // grid_size

#     patches = []
#     for gy in range(grid_size):
#         for gx in range(grid_size):
#             left = gx * patch_w
#             top = gy * patch_h
#             right = w if gx == grid_size - 1 else (left + patch_w)
#             bottom = h if gy == grid_size - 1 else (top + patch_h)

#             patch = img.crop((left, top, right, bottom))
#             patches.append(patch.resize(out_size, resample=Image.BICUBIC))

#     return patches


# def center_saliency_crops(img: Image.Image, out_size=(224, 224)) -> list[Image.Image]:
#     """
#     Lightweight 'saliency-like' center crops: smaller crops around the center.
#     Not true saliency, but emphasizes likely subject region.
#     """
#     w, h = img.size
#     crops = []

#     for scale in [0.7, 0.5]:
#         cw, ch = int(w * scale), int(h * scale)
#         left = (w - cw) // 2
#         top = (h - ch) // 2
#         crop = img.crop((left, top, left + cw, top + ch))
#         crops.append(crop.resize(out_size, resample=Image.BICUBIC))

#     return crops


# def get_image_embedding(
#     img: Image.Image,
#     processor: CLIPProcessor,
#     model: CLIPModel,
#     temp: float = 0.7,
#     out_size=(224, 224)
# ) -> torch.Tensor:
#     """
#     Build a rich multi-view embedding:
#       - Strong TTA views
#       - Multi-crops (center + corners + zoomed)
#       - Grid patches
#       - Center-focused crops

#     Then:
#       - Batch all views through CLIP
#       - Average features
#       - Normalize + temperature scale
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     # Collect all views
#     views = []
#     views.extend(generate_tta_views(img, num_random_augs=3, out_size=out_size))
#     views.extend(multi_crops(img, out_size=out_size))
#     views.extend(grid_patches(img, grid_size=3, out_size=out_size))
#     views.extend(center_saliency_crops(img, out_size=out_size))

#     with torch.no_grad():
#         inputs = processor(images=views, return_tensors="pt")
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         # Autocast for speed if on GPU
#         use_amp = isinstance(device, torch.device) and (device.type == "cuda")
#         if hasattr(torch.cuda, "amp"):
#             ctx = torch.cuda.amp.autocast(enabled=use_amp)
#         else:
#             class DummyCtx:
#                 def __enter__(self): return None
#                 def __exit__(self, exc_type, exc_val, exc_tb): return False
#             ctx = DummyCtx()

#         with ctx:
#             feats = model.get_image_features(**inputs)  # (V, d)

#         feats = F.normalize(feats, p=2, dim=-1)
#         feat_mean = feats.mean(dim=0)  # (d,)

#         # Temperature scaling sharpens similarity
#         feat_scaled = feat_mean / temp
#         feat_final = F.normalize(feat_scaled, p=2, dim=-1)

#     return feat_final


# # ============================================================
# #                 IMAGE SELECTION
# # ============================================================

# def choose_image(
#     target,
#     sentence,
#     images,
#     image_dict,
#     model=None,
#     processor=None,
#     print_output=False
# ):
#     """
#     Final, maximally-performing choose_image():

#       - Text:
#         * Lightweight multi-prompt CLIP text embedding
#         * No WordNet, no NER, no complex semantics
#       - Images:
#         * Rich multi-view CLIP image embeddings (TTA + crops + patches)

#       Returns:
#         ranked_images, ranked_captions (None), ranked_embs (None)
#     """
#     if not (isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel)):
#         raise ValueError("Processor and model must be CLIPProcessor and CLIPModel.")

#     # 1. Build final text embedding using simple prompts
#     text_emb = build_text_embedding(target, sentence, processor, model)
#     text_emb = text_emb.unsqueeze(0)  # (1, d)
#     device = next(model.parameters()).device

#     # 2. Compute image embeddings
#     img_embs = []
#     valid_names = []

#     for name in images:
#         valid_names.append(name)

#         if name not in image_dict:
#             d = model.config.projection_dim
#             img_embs.append(torch.zeros(d, device=device))
#             continue

#         img = image_dict[name]
#         emb = get_image_embedding(img, processor=processor, model=model, temp=0.7)
#         img_embs.append(emb)

#     img_feats = torch.stack(img_embs, dim=0)  # (N, d)

#     # 3. Rank by cosine similarity (dot, since everything is normalized)
#     sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
#     ranked_idx = np.argsort(sims)[::-1]
#     ranked_imgs = [valid_names[i] for i in ranked_idx]

#     if print_output:
#         for rank, i in enumerate(ranked_idx):
#             name = valid_names[i]
#             if name in image_dict:
#                 plt.imshow(image_dict[name])
#                 plt.title(f"Rank {rank+1} | sim={sims[i]:.4f}\nSentence: {sentence}")
#                 plt.axis("off")
#                 plt.show()
#         print("Ranked Images:", ranked_imgs)

#     ranked_captions = [None] * len(ranked_imgs)
#     ranked_embs = [None] * len(ranked_imgs)

#     return ranked_imgs, ranked_captions, ranked_embs


# # ============================================================
# #                 MAIN EVALUATION LOOP
# # ============================================================

# if __name__ == "__main__":

#     # Device selection
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     file_path = "dataset"
#     print_output = False  # Set True to visualize

#     model_name = "openai/clip-vit-base-patch32"
#     processor = CLIPProcessor.from_pretrained(model_name)
#     model = CLIPModel.from_pretrained(model_name).to(device)

#     # Load data
#     data, image_dict = load_data(file_path=file_path, train_val="trial")

#     predicted_ranks = []
#     for idx, row in data.iterrows():
#         target = row["target"]
#         sentence = row["sentence"]
#         images = [row[f"image_{i}"] for i in range(10)]
#         label = row["label"]

#         ranked_images, _, _ = choose_image(
#             target,
#             sentence,
#             images,
#             image_dict,
#             model=model,
#             processor=processor,
#             print_output=print_output,
#         )

#         predicted_rank = ranked_images.index(label) + 1
#         print("Predicted Rank:", predicted_rank)
#         predicted_ranks.append(predicted_rank)

#     predicted_ranks = np.array(predicted_ranks)
#     mrr = np.mean(1.0 / predicted_ranks)
#     hit_rate = np.mean(predicted_ranks == 1)

#     print("---------------------------------")
#     print(f"MRR: {mrr}")
#     print(f"Hit Rate: {hit_rate}")
# import os
# import random
# import torch
# import pandas as pd
# import numpy as np
# from PIL import Image, ImageFilter, ImageEnhance
# import matplotlib.pyplot as plt

# from tqdm import tqdm
# from transformers import CLIPModel, CLIPProcessor
# import torch.nn.functional as F
# import spacy
# from nltk.corpus import wordnet as wn


# ##############################
# # Load SemEval Data
# ##############################

# def load_data(file_path, train_val="test", target_size=(384, 384)):
#     """
#     Load the SemEval dataset.

#     Expected structure:
#         <file_path>/
#             train_v1/
#                 train.data.v1.txt
#                 train.gold.v1.txt
#                 train_images_v1/
#             trial_v1/
#             test_v1/
#     """
#     path = os.path.join(file_path, train_val + "_v1")

#     # Load text data
#     path_data = os.path.join(path, train_val + ".data.v1.txt")
#     data = pd.read_csv(path_data, sep='\t', header=None)
#     data.columns = ['target', 'sentence'] + [
#         f'image_{i}' for i in range(data.shape[1] - 2)
#     ]

#     # Load labels
#     path_labels = os.path.join(path, train_val + ".gold.v1.txt")
#     with open(path_labels, "r") as f:
#         gold_labels = [line.strip() for line in f]
#     data['label'] = gold_labels

#     # Load images
#     path_images = os.path.join(path, train_val + "_images_v1")
#     image_dict = {}
#     files = os.listdir(path_images)

#     for filename in tqdm(files, total=len(files), desc="Loading Images", unit="image"):
#         if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#             try:
#                 img = Image.open(os.path.join(path_images, filename)).convert('RGB')
#                 image_dict[filename] = img
#             except Exception:
#                 continue

#     return data, image_dict


# ##############################
# # Basic CLIP Text Embeddings
# ##############################

# def get_clip_text_emb(text, processor, model):
#     """
#     Single text → CLIP embedding (normalized).
#     """
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         feat = model.get_text_features(**inputs)
#         feat = F.normalize(feat, p=2, dim=-1)
#     return feat[0]   # (d,)


# def get_clip_text_emb_from_list(text_list, processor, model):
#     """
#     Multiple texts → average CLIP embedding (normalized).
#     """
#     device = next(model.parameters()).device
#     embs = []

#     with torch.no_grad():
#         for text in text_list:
#             inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             feat = model.get_text_features(**inputs)
#             feat = F.normalize(feat, p=2, dim=-1)
#             embs.append(feat[0])

#     embs = torch.stack(embs, dim=0)        # (N, d)
#     mean_emb = embs.mean(dim=0)            # (d,)
#     mean_emb = F.normalize(mean_emb, p=2, dim=-1)
#     return mean_emb


# ##############################
# # Context Word Extraction (spaCy)
# ##############################

# def extract_context_word(target: str, sentence: str) -> str:
#     """
#     Try to extract the 'context word' when the sentence looks like:
#       - 'bank erosion'
#       - 'internet router'
#       - 'andromeda tree'
#     If that fails, fall back to the full sentence as context.
#     """
#     if sentence is None:
#         return ""

#     s = str(sentence).strip()
#     t = str(target).strip()

#     if not s:
#         return ""

#     tokens = s.split()

#     # Case 1: exactly two tokens, one of them is the target
#     if len(tokens) == 2:
#         t_low = t.lower()
#         tok0, tok1 = tokens[0], tokens[1]
#         if tok0.lower() == t_low:
#             return tok1
#         if tok1.lower() == t_low:
#             return tok0

#     # Otherwise, just treat the whole sentence as "context"
#     return s

# def get_synonym(word: str) -> str | None:
#     """
#     Very lightweight WordNet synonym helper.
#     Returns a single short synonym (or None if not available).
#     """
#     if word is None:
#         return None

#     w = str(word).strip()
#     if not w:
#         return None

#     synsets = wn.synsets(w)
#     if not synsets:
#         return None

#     lemmas = synsets[0].lemma_names()
#     if not lemmas:
#         return None

#     return lemmas[0].replace("_", " ")

# def build_micro_prompts(target: str, sentence: str) -> list[str]:
#     """
#     Build a set of short, CLIP-friendly micro-prompts
#     from the target word and its context phrase.
#     """
#     target = str(target).strip()
#     sentence = "" if sentence is None else str(sentence).strip()

#     # Core context word/phrase (e.g. 'erosion' from 'bank erosion')
#     context = extract_context_word(target, sentence).strip()
#     base_context = context if context else sentence

#     prompts: list[str] = []

#     # 1–11: core micro-prompts
#     prompts.append(f"{target} {base_context}".strip())
#     prompts.append(f"{base_context} {target}".strip())
#     prompts.append(f"{target} {base_context} photo".strip())
#     prompts.append(f"{target} {base_context} scene".strip())
#     prompts.append(f"{target} near {base_context}".strip())
#     prompts.append(f"{target} with {base_context}".strip())
#     prompts.append(f"a photo of {target} in {base_context}".strip())
#     prompts.append(f"real {target} with {base_context}".strip())
#     prompts.append(f"{target} related to {base_context}".strip())
#     prompts.append(f"{target} appearing in a {base_context} setting".strip())
#     prompts.append(f"visual sense of {target} in {base_context}".strip())

#     # 12: synonym-expanded variant
#     target_syn = get_synonym(target)
#     context_syn = get_synonym(context) if context else None

#     syn_target = target_syn if target_syn else target
#     syn_context = context_syn if context_syn else base_context

#     if syn_target and syn_context:
#         prompts.append(f"{syn_target} {syn_context}".strip())

#     # Remove any accidental empties
#     prompts = [p for p in prompts if p]

#     return prompts

# ##############################
# # Build Final Text Embedding (Micro-prompt Fusion)
# ##############################

# # def build_text_embedding(
# #     target,
# #     sentence,
# #     processor,
# #     model,
# #     nlp,
# #     weights=(0.60, 0.25, 0.15),
# # ):
# #     """
# #     final_text_emb = w1 * sentence_emb
# #                    + w2 * target_context_emb
# #                    + w3 * micro_prompt_emb

# #     where:
# #       - sentence_emb: CLIP(text=sentence)
# #       - target_context_emb: CLIP(text="target context_word")
# #       - micro_prompt_emb: mean CLIP(text of short prompts)
# #     """
# #     # 1. Sentence embedding
# #     sentence_emb = get_clip_text_emb(sentence, processor, model)   # (d,)

# #     # 2. Extract context word
# #     context_word = extract_context_word(sentence, target, nlp)
# #     if context_word is not None:
# #         pair_text = f"{target} {context_word}"
# #     else:
# #         pair_text = target

# #     target_context_emb = get_clip_text_emb(pair_text, processor, model)

# #     # 3. Micro-prompts
# #     if context_word is not None:
# #         prompts = [
# #             f"{context_word} {target}",
# #             f"{target} {context_word}",
# #             f"{target} near {context_word}",
# #             f"{target} related to {context_word}",
# #             f"{target} by the {context_word}",
# #             f"photo of {target}",
# #         ]
# #     else:
# #         # no reliable context → keep prompts short
# #         prompts = [
# #             target,
# #             f"{context_word} {target}",
# #             f"a photo of {target} by {context_word}",
# #             f"photo of {target}",
# #             f"{target} object",
# #         ]

# #     micro_prompt_emb = get_clip_text_emb_from_list(prompts, processor, model)

# #     # 4. Weighted fusion
# #     w_sent, w_pair, w_prompt = weights
# #     fused = (
# #         w_sent   * sentence_emb +
# #         w_pair   * target_context_emb +
# #         w_prompt * micro_prompt_emb
# #     )
# #     fused = F.normalize(fused, p=2, dim=-1)      # (d,)
# #     return fused.unsqueeze(0)                    # (1, d)

# def get_text_embedding_from_micro_prompts(
#     target: str,
#     sentence: str,
#     processor: CLIPProcessor,
#     model: CLIPModel,
# ) -> torch.Tensor:
#     """
#     Encode 12+ micro-prompts with CLIP text encoder,
#     average them, and L2-normalize to get a strong text embedding.
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     prompts = build_micro_prompts(target, sentence)

#     with torch.no_grad():
#         inputs = processor(
#             text=prompts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         feats = model.get_text_features(**inputs)   # (P, d)
#         feats = F.normalize(feats, p=2, dim=-1)
#         mean_feat = feats.mean(dim=0)              # (d,)
#         mean_feat = F.normalize(mean_feat, p=2, dim=-1)

#     return mean_feat   # shape (d,)

# ##############################
# # Image Augmentations (your strong pipeline)
# ##############################

# def random_geometric_augment(img: Image.Image) -> Image.Image:
#     """Mild random geometric transforms: flip, rotation, slight zoom/crop."""
#     w, h = img.size
#     out = img

#     # Random horizontal flip (50% chance)
#     if random.random() < 0.5:
#         out = out.transpose(Image.FLIP_LEFT_RIGHT)

#     # Small random rotation
#     angle = random.uniform(-7, 7)  # degrees
#     out = out.rotate(angle, resample=Image.BICUBIC, expand=False)

#     # Slight random crop (95–100% of size)
#     scale = random.uniform(0.95, 1.0)
#     new_w, new_h = int(w * scale), int(h * scale)
#     if new_w < w and new_h < h:
#         left = random.randint(0, w - new_w)
#         top = random.randint(0, h - new_h)
#         out = out.crop((left, top, left + new_w, top + new_h))
#         out = out.resize((w, h), resample=Image.BICUBIC)

#     return out


# def random_photometric_augment(img: Image.Image) -> Image.Image:
#     """Mild brightness/contrast/color/blur jitter."""
#     out = img

#     # Brightness
#     if random.random() < 0.5:
#         enhancer = ImageEnhance.Brightness(out)
#         factor = random.uniform(0.95, 1.05)
#         out = enhancer.enhance(factor)

#     # Contrast
#     if random.random() < 0.5:
#         enhancer = ImageEnhance.Contrast(out)
#         factor = random.uniform(0.95, 1.05)
#         out = enhancer.enhance(factor)

#     # Color
#     if random.random() < 0.5:
#         enhancer = ImageEnhance.Color(out)
#         factor = random.uniform(0.95, 1.05)
#         out = enhancer.enhance(factor)

#     # Slight blur (very mild)
#     if random.random() < 0.3:
#         out = out.filter(ImageFilter.GaussianBlur(radius=0.3))

#     return out


# def generate_tta_views(img: Image.Image,
#                        num_random_augs: int = 2,
#                        out_size=(224, 224)) -> list[Image.Image]:
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


# def multi_crops(img: Image.Image, out_size=(224, 224)) -> list[Image.Image]:
#     """Center + four corners + zoomed center (multi-crop)."""
#     w, h = img.size
#     crops = []

#     # Main center crop (80%)
#     crop_w, crop_h = int(w * 0.8), int(h * 0.8)
#     crop_w = max(1, crop_w)
#     crop_h = max(1, crop_h)

#     # Center
#     left = (w - crop_w) // 2
#     top = (h - crop_h) // 2
#     center = img.crop((left, top, left + crop_w, top + crop_h))
#     crops.append(center.resize(out_size, resample=Image.BICUBIC))

#     # Top-left
#     tl = img.crop((0, 0, crop_w, crop_h))
#     crops.append(tl.resize(out_size, resample=Image.BICUBIC))

#     # Top-right
#     tr = img.crop((w - crop_w, 0, w, crop_h))
#     crops.append(tr.resize(out_size, resample=Image.BICUBIC))

#     # Bottom-left
#     bl = img.crop((0, h - crop_h, crop_w, h))
#     crops.append(bl.resize(out_size, resample=Image.BICUBIC))

#     # Bottom-right
#     br = img.crop((w - crop_w, h - crop_h, w, h))
#     crops.append(br.resize(out_size, resample=Image.BICUBIC))

#     # Zoomed-in center (60%)
#     zoom_scale = 0.6
#     z_w, z_h = int(w * zoom_scale), int(h * zoom_scale)
#     z_left = (w - z_w) // 2
#     z_top = (h - z_h) // 2
#     zoom_center = img.crop((z_left, z_top, z_left + z_w, z_top + z_h))
#     crops.append(zoom_center.resize(out_size, resample=Image.BICUBIC))

#     return crops


# def grid_patches(img: Image.Image, grid_size=3, out_size=(224, 224)) -> list[Image.Image]:
#     """Split the image into a grid (e.g., 3x3) and return patches."""
#     w, h = img.size
#     patch_w = w // grid_size
#     patch_h = h // grid_size

#     patches = []
#     for gy in range(grid_size):
#         for gx in range(grid_size):
#             left = gx * patch_w
#             top = gy * patch_h
#             right = w if gx == grid_size - 1 else (left + patch_w)
#             bottom = h if gy == grid_size - 1 else (top + patch_h)

#             patch = img.crop((left, top, right, bottom))
#             patches.append(patch.resize(out_size, resample=Image.BICUBIC))

#     return patches


# def center_saliency_crops(img: Image.Image, out_size=(224, 224)) -> list[Image.Image]:
#     """
#     Lightweight 'saliency-like' center crops: smaller crops around the center.
#     Not true saliency, but emphasizes likely subject region.
#     """
#     w, h = img.size
#     crops = []

#     for scale in [0.7, 0.5]:
#         cw, ch = int(w * scale), int(h * scale)
#         left = (w - cw) // 2
#         top = (h - ch) // 2
#         crop = img.crop((left, top, left + cw, top + ch))
#         crops.append(crop.resize(out_size, resample=Image.BICUBIC))

#     return crops


# def get_image_embedding(
#     img: Image.Image,
#     processor: CLIPProcessor,
#     model: CLIPModel,
#     temp: float = 0.65,
#     out_size=(224, 224)
# ) -> torch.Tensor:
#     """
#     Build a rich multi-view embedding:
#     - Strong TTA views
#     - Multi-crops (center + corners + zoomed)
#     - Grid patches
#     - Center-focused crops

#     Then:
#     - Batch all views through CLIP
#     - Average features
#     - Normalize + temperature scale
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     # Collect all views
#     views = []
#     views.extend(generate_tta_views(img, num_random_augs=2, out_size=out_size))
#     views.extend(multi_crops(img, out_size=out_size))
#     views.extend(grid_patches(img, grid_size=3, out_size=out_size))
#     views.extend(center_saliency_crops(img, out_size=out_size))

#     with torch.no_grad():
#         inputs = processor(images=views, return_tensors="pt")
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         use_amp = (isinstance(device, torch.device) and device.type == "cuda")
#         if hasattr(torch.cuda, "amp"):
#             ctx = torch.cuda.amp.autocast(enabled=use_amp)
#         else:
#             class DummyCtx:
#                 def __enter__(self): return None
#                 def __exit__(self, exc_type, exc_val, exc_tb): return False
#             ctx = DummyCtx()

#         with ctx:
#             feats = model.get_image_features(**inputs)  # (N, d)

#         feats = F.normalize(feats, p=2, dim=-1)
#         feat_mean = feats.mean(dim=0)  # (d,)

#         # Temperature scaling sharpens cosine similarities
#         feat_scaled = feat_mean / temp
#         feat_final = F.normalize(feat_scaled, p=2, dim=-1)

#     return feat_final


# ##############################
# # Main Matching Function
# ##############################

# def choose_image(
#     target,
#     sentence,
#     images,
#     image_dict,
#     model,
#     processor,
#     nlp,
#     text_weights=(0.60, 0.25, 0.15),
#     img_temp=0.65,
#     print_output=False
# ):
#     """
#     - Build fused text embedding:
#         sentence + (target, context_word) + micro-prompts
#     - Build strong image embeddings via multi-view CLIP
#     - Rank images by cosine similarity
#     """
#     device = next(model.parameters()).device

#     # 1. Build final text embedding
#     text_emb = get_text_embedding_from_micro_prompts(
#         target=target,
#         sentence=sentence,
#         processor=processor,
#         model=model,
#         # nlp=nlp,
#         # weights=text_weights,
#     )  # (1, d)

#     # 2. Compute image embeddings
#     img_embs = []
#     valid_names = []

#     for name in images:
#         valid_names.append(name)

#         if name not in image_dict:
#             d = model.config.projection_dim
#             img_embs.append(torch.zeros(d, device=device))
#             continue

#         img = image_dict[name]
#         emb = get_image_embedding(
#             img,
#             processor=processor,
#             model=model,
#             temp=img_temp
#         )
#         img_embs.append(emb)

#     img_feats = torch.stack(img_embs, dim=0)  # (N, d)

#     # 3. Rank
#     sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
#     ranked_idx = np.argsort(sims)[::-1]
#     ranked_imgs = [valid_names[i] for i in ranked_idx]

#     if print_output:
#         print(f"\nSentence: {sentence}")
#         print(f"Target: {target}")
#         print("Ranked images (best → worst):", ranked_imgs)

#     ranked_captions = [None] * len(ranked_imgs)
#     ranked_embs = [None] * len(ranked_imgs)

#     return ranked_imgs, ranked_captions, ranked_embs


# ##############################
# # Main Script
# ##############################

# if __name__ == "__main__":

#     # Device
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     file_path = "dataset"
#     print_output = False

#     # CLIP model
#     model_name = "openai/clip-vit-base-patch32"
#     processor = CLIPProcessor.from_pretrained(model_name)
#     model = CLIPModel.from_pretrained(model_name).to(device)

#     # spaCy (for context word extraction)
#     nlp = spacy.load("en_core_web_sm")

#     # Load data
#     data, image_dict = load_data(file_path=file_path, train_val="trial")

#     predicted_ranks = []
#     for idx, row in data.iterrows():
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
#             nlp=nlp,
#             text_weights=(0.60, 0.25, 0.15),  # can tune
#             img_temp=0.65,                     # can tune
#             print_output=print_output
#         )

#         predicted_rank = ranked_images.index(label) + 1
#         print("Predicted Rank:", predicted_rank)
#         predicted_ranks.append(predicted_rank)

#     predicted_ranks = np.array(predicted_ranks)
#     mrr = np.mean(1.0 / predicted_ranks)
#     hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)

#     print("---------------------------------")
#     print(f"MRR: {mrr}")
#     print(f"Hit Rate: {hit_rate}")
