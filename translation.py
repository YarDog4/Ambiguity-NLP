import os
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, MarianMTModel, MarianTokenizer
import torch.nn.functional as F
from nltk.corpus import wordnet as wn


############################## Load in the SemEval data ##############################

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


############################## Translation Functions ##############################

def get_translator(source_lang="en", target_lang="es"):
    """Load translation model and tokenizer"""
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer


############################## Advanced Prompting Functions ##############################

def context_window(sentence: str, target: str, window=5):
    """Extract context window around target word"""
    toks = sentence.split()
    try:
        i = toks.index(target)
    except ValueError:
        return sentence
    lo, hi = max(0, i - window), min(len(toks), i + window + 1)
    return " ".join(toks[lo:hi])

def build_text_prompts(target: str, sentence: str):
    """Builds multiple contextual sentence prompts"""
    ctx = context_window(sentence, target, window=5)
    return [
        sentence,
        ctx,
        f"In this sentence, the word '{target}' refers to the correct image: {ctx}",
        f"A picture that matches the sense of '{target}' in: {ctx}",
        f"Focus on the meaning of '{target}' here: {ctx}",
    ]

def synonym_prompts(target):
    """Generates synonym-based prompts using WordNet"""
    syns = []
    for syn in wn.synsets(target):
        name = syn.name().split('.')[0].replace('_',' ')
        definition = syn.definition()
        syns.append(f"a photo of {name}")
        syns.append(f"an image showing {definition}")
        if len(syns) >= 5:
            break
    return syns[:5]

def build_target_only_prompts(target):
    """Combine basic 'photo of' prompts with WordNet synonyms"""
    base = [f"a photo of {target}", f"an image of {target}", f"{target}"]
    try:
        return base + synonym_prompts(target)
    except:
        return base

def sharp_prompts(target, sentence):
    """Sharp prompts for re-ranking"""
    return [
        f"the intended meaning of '{target}' in: {sentence}",
        f"correct interpretation of '{target}' here: {sentence}",
        f"what '{target}' means in context: {sentence}",
    ]


############################## Multilingual Prompting Functions ##############################

def translate_prompts(prompts, trans_model, trans_tokenizer, device, print_translations=False):
    """Translate a list of prompts to target language"""
    translated = []
    for prompt in prompts:
        inputs = trans_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            translated_tokens = trans_model.generate(**inputs)
        trans_text = trans_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated.append(trans_text)
        if print_translations:
            print(f"    EN: {prompt}")
            print(f"    ->: {trans_text}")
    return translated

def build_multilingual_prompts(target, sentence, prompt_type, languages, device, print_translations=False):
    """Build prompts in multiple languages
    
    Args:
        target (str): Target word
        sentence (str): Sentence
        prompt_type (str): 'text', 'target', or 'sharp'
        languages (list): List of language codes
        device: Device for translation models
        print_translations (bool): Whether to print translations
        
    Returns:
        all_prompts (list): Combined prompts from all languages
    """
    # English prompts
    if prompt_type == 'text':
        en_prompts = build_text_prompts(target, sentence)
    elif prompt_type == 'target':
        en_prompts = build_target_only_prompts(target)
    elif prompt_type == 'sharp':
        en_prompts = sharp_prompts(target, sentence)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")
    
    all_prompts = en_prompts.copy()
    
    if print_translations:
        print(f"\n  [{prompt_type.upper()} PROMPTS - English]")
        for p in en_prompts:
            print(f"    {p}")
    
    # Add translations
    translation_cache = {}
    for lang in languages:
        try:
            # Get or load translation model
            if lang not in translation_cache:
                trans_model, trans_tokenizer = get_translator("en", lang)
                trans_model = trans_model.to(device)
                translation_cache[lang] = (trans_model, trans_tokenizer)
            else:
                trans_model, trans_tokenizer = translation_cache[lang]
            
            if print_translations:
                print(f"\n  [{prompt_type.upper()} PROMPTS - {lang.upper()}]")
            
            # Translate prompts
            translated = translate_prompts(en_prompts, trans_model, trans_tokenizer, device, print_translations)
            all_prompts.extend(translated)
            
        except Exception as e:
            print(f"  Warning: Could not translate {prompt_type} prompts to {lang}: {e}")
            continue
    
    return all_prompts


############################## Get Embeddings ##############################

def get_sentence_embedding(text, tokenizer=None, model=None):
    """Get the sentence embedding"""
    if isinstance(tokenizer, CLIPProcessor) and isinstance(model, CLIPModel):
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = tokenizer(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_text_features(**inputs)
            features = F.normalize(features, p=2, dim=-1)
        return features[0]

@torch.no_grad()
def get_clip_text_embedding_multi(prompts, processor: CLIPProcessor, model: CLIPModel):
    """Computes and averages CLIP text embeddings for multiple prompts"""
    device = next(model.parameters()).device
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_text_features(**inputs)  # (M, d)
    feats = F.normalize(feats, p=2, dim=-1)
    mean = feats.mean(dim=0, keepdim=True)
    mean = F.normalize(mean, p=2, dim=-1)  # (1, d)
    return mean.squeeze(0)  # (d,)

def get_blended_text_embedding(target, sentence, processor, model, alpha=0.7, 
                                use_translation=False, languages=None, 
                                print_translations=False):
    """Creates balanced CLIP text embedding with optional multilingual support
    
    Args:
        target (str): Target word
        sentence (str): Sentence
        processor: CLIP processor
        model: CLIP model
        alpha (float): Weight for sentence vs target (0-1)
        use_translation (bool): Whether to use multilingual prompts
        languages (list): List of language codes
        print_translations (bool): Whether to print translations
        
    Returns:
        blended embedding (Tensor)
    """
    device = next(model.parameters()).device
    
    if use_translation and languages:
        # Build multilingual prompts
        sent_prompts = build_multilingual_prompts(target, sentence, 'text', languages, device, print_translations)
        tgt_prompts = build_multilingual_prompts(target, sentence, 'target', languages, device, print_translations)
    else:
        # English only prompts
        sent_prompts = build_text_prompts(target, sentence)
        tgt_prompts = build_target_only_prompts(target)
        if print_translations:
            print(f"\n  [TEXT PROMPTS - English]")
            for p in sent_prompts:
                print(f"    {p}")
            print(f"\n  [TARGET PROMPTS - English]")
            for p in tgt_prompts:
                print(f"    {p}")
    
    sent_emb = get_clip_text_embedding_multi(sent_prompts, processor, model)
    tgt_emb = get_clip_text_embedding_multi(tgt_prompts, processor, model)
    
    blend = F.normalize(alpha * sent_emb + (1 - alpha) * tgt_emb, p=2, dim=-1)
    return blend.unsqueeze(0)

def clip_image_feats_with_tta(pil_list, processor, model):
    """Compute averaged CLIP image features with TTA (original + horizontal flip)"""
    device = next(model.parameters()).device
    flipped_imgs = [ImageOps.mirror(img) for img in pil_list]
    aug_batches = [pil_list, flipped_imgs]
    feats_accum = None

    for imgs in aug_batches:
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, p=2, dim=-1)
        feats_accum = feats if feats_accum is None else feats_accum + feats

    feats_mean = F.normalize(feats_accum / len(aug_batches), p=2, dim=-1)
    return feats_mean


############################## Image Selection ##############################

def choose_image(target, sentence, images, image_dict, 
                 tokenizer=None, model=None, processor=None, blip_model=None, 
                 use_translation=False, languages=None, 
                 use_rerank=True, print_output=False, 
                 print_translations=False):
    """Choose the best matching image with advanced prompting and optional translation
    
    Args:
        target (str): Target word
        sentence (str): Sentence
        images (list): List of image filenames
        image_dict (dict): Dictionary mapping filenames to PIL images
        tokenizer: Not used (kept for compatibility)
        model: CLIP model
        processor: CLIP processor
        blip_model: Not used (kept for compatibility)
        use_translation (bool): Whether to use multilingual prompts
        languages (list): List of language codes for translation
        use_rerank (bool): Whether to use sharp prompt re-ranking
        print_output (bool): Whether to print debug output
        print_translations (bool): Whether to print all translations
        
    Returns:
        ranked_images (list): Images ranked by similarity
        ranked_captions (list): Placeholder (None)
        ranked_embs (list): Placeholder (None)
    """
    
    if isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel):
        if print_output or print_translations: 
            print("\n" + "="*70)
            print(f"Target: '{target}'")
            print(f"Sentence: {sentence}")
            if use_translation and languages:
                print(f"Using multilingual prompts: {', '.join(languages)}")
        
        device = next(model.parameters()).device
        
        # 1. Get text embeddings with multiple alphas
        alphas = [0.6, 0.8]
        embeddings = []
        for a in alphas:
            emb = get_blended_text_embedding(
                target, sentence, processor, model, alpha=a,
                use_translation=use_translation, languages=languages,
                print_translations=print_translations
            )
            embeddings.append(emb)
        
        # 2. Average and normalize
        text_emb = F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=-1)  # (1, d)
        
        # 3. Get image embeddings with TTA
        pil_batch = []
        valid_names = []
        for name in images:
            if name in image_dict:
                pil_batch.append(image_dict[name])
                valid_names.append(name)
            else:
                pil_batch.append(Image.new('RGB', (224, 224)))
                valid_names.append(name)
        
        with torch.no_grad():
            img_feats = clip_image_feats_with_tta(pil_batch, processor, model)  # (N, d)
        
        # 4. Compute similarities
        sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()  # (N,)
        
        # 5. Optional re-ranking with sharp prompts
        if use_rerank:
            ranked_indices = np.argsort(sims)[::-1].copy()
            topk = ranked_indices[: min(5, len(ranked_indices))]
            
            # Build sharp prompts (multilingual if enabled)
            if use_translation and languages:
                if print_translations:
                    print("\n[SHARP RE-RANKING PROMPTS]")
                sharp_prompt_list = build_multilingual_prompts(target, sentence, 'sharp', languages, device, print_translations)
            else:
                sharp_prompt_list = sharp_prompts(target, sentence)
                if print_translations:
                    print(f"\n  [SHARP PROMPTS - English]")
                    for p in sharp_prompt_list:
                        print(f"    {p}")
            
            sharp_q = get_clip_text_embedding_multi(sharp_prompt_list, processor, model).unsqueeze(0)
            sharp_sims = (sharp_q @ img_feats[topk].T).squeeze(0).detach().cpu().numpy()
            
            # Blend original and sharp similarities
            beta = 0.5
            sims[topk] = beta * sims[topk] + (1 - beta) * sharp_sims
        
        # 6. Final ranking
        ranked_indices = np.argsort(sims)[::-1]
        ranked_images = [valid_names[int(i)] for i in ranked_indices]

        if print_output:
            print(f"\nTop 3 ranked images:")
            for rank, i in enumerate(ranked_indices[:3]):
                print(f"  Rank {rank+1}: {valid_names[i]} (similarity: {sims[i]:.4f})")

        ranked_captions = [None for _ in ranked_images]
        ranked_embs = [None for _ in ranked_images]

        return ranked_images, ranked_captions, ranked_embs


############################## Main Experiments ##############################

if __name__ == "__main__":
    
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"

    file_path = "dataset"
    print_output = False  # Set to True to see individual examples
    print_translations = True  # Set to True to see all translations

    # Define the various pretrained models
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    # Load in the data
    data, image_dict = load_data(file_path=file_path, train_val="test")

    # Comprehensive experiment configurations
    experiments = [
        # Baseline with advanced prompting (no translation)
        {"name": "Baseline (Advanced Prompting + Rerank)", 
         "use_translation": False, "languages": None, "use_rerank": True},
        
        {"name": "Baseline (Advanced Prompting, No Rerank)", 
         "use_translation": False, "languages": None, "use_rerank": False},
        
        # Single language experiments
        {"name": "Spanish (All Prompts + Rerank)", 
         "use_translation": True, "languages": ["es"], "use_rerank": True},
        
        {"name": "French (All Prompts + Rerank)", 
         "use_translation": True, "languages": ["fr"], "use_rerank": True},
        
        {"name": "German (All Prompts + Rerank)", 
         "use_translation": True, "languages": ["de"], "use_rerank": True},
        
        # Multi-language experiments with reranking
        {"name": "Multi ES+FR (All Prompts + Rerank)", 
         "use_translation": True, "languages": ["es", "fr"], "use_rerank": True},
        
        {"name": "Multi ES+DE (All Prompts + Rerank)", 
         "use_translation": True, "languages": ["es", "de"], "use_rerank": True},
        
        {"name": "Multi FR+DE (All Prompts + Rerank)", 
         "use_translation": True, "languages": ["fr", "de"], "use_rerank": True},
        
        {"name": "Multi ES+FR+DE (All Prompts + Rerank)", 
         "use_translation": True, "languages": ["es", "fr", "de"], "use_rerank": True},
        
        # Without re-ranking
        {"name": "Multi ES+FR (No Rerank)", 
         "use_translation": True, "languages": ["es", "fr"], "use_rerank": False},
        
        {"name": "Multi ES+FR+DE (No Rerank)", 
         "use_translation": True, "languages": ["es", "fr", "de"], "use_rerank": False},
    ]

    results = []
    
    for exp in experiments:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {exp['name']}")
        print(f"{'='*70}")
        
        predicted_ranks = []
        
        # Only show translations for first example of each experiment
        show_first = print_translations
        
        for idx, row in data.iterrows():
            target = row['target']
            sentence = row['sentence']
            images = [row[f'image_{i}'] for i in range(10)]
            label = row['label']

            if label not in image_dict:
                print(f"Skipping {label} (not found)")
                continue

            # Show translations only for first example
            current_print_translations = show_first and idx == 0
            
            ranked_images, _, _ = choose_image(
                target, sentence, images, image_dict,
                tokenizer=None, model=model, processor=tokenizer, blip_model=None,
                use_translation=exp["use_translation"],
                languages=exp["languages"],
                use_rerank=exp["use_rerank"],
                print_output=print_output,
                print_translations=current_print_translations
            )
            
            predicted_rank = ranked_images.index(label) + 1
            predicted_ranks.append(predicted_rank)

        predicted_ranks = np.array(predicted_ranks)
        mrr = np.mean(1/predicted_ranks)
        hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)
        
        results.append({
            "experiment": exp["name"],
            "mrr": mrr,
            "hit_rate": hit_rate
        })
        
        print(f"\nResults for {exp['name']}:")
        print(f"  MRR: {mrr:.4f}")
        print(f"  Hit Rate: {hit_rate:.4f}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON - Advanced Prompting + Translation")
    print(f"{'='*70}")
    print(f"{'Experiment':<50} {'MRR':<12} {'Hit Rate':<12}")
    print("-" * 74)
    for r in results:
        print(f"{r['experiment']:<50} {r['mrr']:<12.4f} {r['hit_rate']:<12.4f}")