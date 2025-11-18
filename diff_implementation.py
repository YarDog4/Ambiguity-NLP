import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageOps
from nltk.corpus import wordnet as wn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
plt.ioff()
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F


############################## Load in the SemEval data ##############################
def load_data(file_path, train_val="train", target_size=(384, 384)):
    """Load in the data

    Args:
        file_path (str): The file path
        train_val (str): Whether to load in the train, test, or trial set
        target_size (tuple): The size of each image to use

    Returns:
        data (DataFrame): Target, Sentence, image_0-9, label
        image_dict (dict): Map image name to image
    """
    # Train/trial/test set
    path = os.path.join(file_path, train_val+"_v1")

    # Load in the data
    path_data = os.path.join(path, train_val+".data.v1.txt")
    data = pd.read_csv(path_data, sep='\t', header=None)
    data.columns = ['target', 'sentence'] + [f'image_{i}' for i in range(data.shape[1] - 2)]  # Rename columns for clarity

    # Load in the labels
    path_labels = os.path.join(path, train_val+".gold.v1.txt")
    with open(path_labels, "r") as f: gold_labels = [line.strip() for line in f]  # Load gold labels as a list
    data['label'] = gold_labels  # Adds as the last column

    # Load in the images
    path_images = os.path.join(path, train_val+"_images_v1")
    image_dict = {}
    files = os.listdir(path_images)
    for filename in tqdm(files, total=len(files), desc="Loading in the Images", unit="image"):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'): 
            try:
                img = Image.open(os.path.join(path_images, filename)).convert('RGB')
                # img_resized = img.resize(target_size, resample=Image.BICUBIC)
                image_dict[filename] = img
            except Exception:
                continue

    return data, image_dict


############################## Get Embeddings ##############################
def get_sentence_embedding(text, tokenizer=None, model=None):   # Get an embedding for a definition or sentence
    """Get the sentence embedding by mean-pooling the last hidden states from BERT
    
    If (tokenizer is CLipProc) and (model is ClipModel), reutnr a normalized clip text embeding in the image-text space. 
    
    Args:
        text (str): The input text
        tokenizer: The tokenizer
        model: The model
        
    Returns:
        embedding (Tensor): The output embedding
    """
    
    if isinstance(tokenizer, CLIPProcessor) and isinstance(model, CLIPModel):
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = tokenizer(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_text_features(**inputs) #(1, d)
            features = F.normalize(features, p=2, dim=-1)
        return features[0]

    # # Fallback
    # if tokenizer is None: tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Initialize tokenizer and model
    # if model is None: model = BertModel.from_pretrained('bert-base-uncased')
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
        last_hidden = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
        embedding = last_hidden.mean(dim=1)[0]  # Mean-pool across all tokens (dim=1)
    return embedding

# Instead of embedding the whole sentence once, build several prompts that explicitly highlight the ambiguous target. 
# Average their normalized CLIP text embeddings.
def context_window(sentence: str, target: str, window=5):
    toks = sentence.split()
    try:
        i = toks.index(target)
    except ValueError:
        return sentence
    lo, hi = max(0, i - window), min(len(toks), i + window + 1)
    return " ".join(toks[lo:hi])

# Builds multiple contextual sentence prompts that highlight the target word for CLIP text embedding.
def build_text_prompts(target: str, sentence: str):
    ctx = context_window(sentence, target, window=5)
    return [
        sentence,
        ctx,
        f"In this sentence, the word '{target}' refers to the correct image: {ctx}",
        f"A picture that matches the sense of '{target}' in: {ctx}",
        f"Focus on the meaning of '{target}' here: {ctx}",
    ]

# Generates up to five synonym-based prompts using WordNet to capture related meanings of the target word.
def synonym_prompts(target):
    syns = []
    for syn in wn.synsets(target):
        name = syn.name().split('.')[0].replace('_',' ')
        definition = syn.definition()
        syns.append(f"a photo of {name}")
        syns.append(f"an image showing {definition}")
        if len(syns) >= 5:
            break
    return syns[:5]

# Combines simple “photo of” prompts with WordNet synonym prompts to expand the target’s semantic coverage.
def build_target_only_prompts(target):
    """Combine basic 'photo of' prompts with WordNet synonyms."""
    base = [f"a photo of {target}", f"an image of {target}", f"{target}"]
    try:
        return base + synonym_prompts(target)
    except:
        return base

# Computes and averages CLIP text embeddings for multiple prompts, producing one normalized embedding vector.  
@torch.no_grad()
def get_clip_text_embedding_multi(prompts, processor: CLIPProcessor, model: CLIPModel):
    device = next(model.parameters()).device
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_text_features(**inputs)  # (M, d)
    feats = F.normalize(feats, p=2, dim=-1)
    mean = feats.mean(dim=0, keepdim=True)
    mean = F.normalize(mean, p=2, dim=-1)  # (1, d)
    return mean.squeeze(0)  # (d,)

# Creates a balanced CLIP text embedding by blending sentence-level and target-only embeddings using a weighting factor α.
def get_blended_text_embedding(target, sentence, processor, model, alpha=0.7):
    sent_emb = get_clip_text_embedding_multi(build_text_prompts(target, sentence), processor, model)
    tgt_emb  = get_clip_text_embedding_multi(build_target_only_prompts(target), processor, model)
    blend = F.normalize(alpha * sent_emb + (1 - alpha) * tgt_emb, p=2, dim=-1)
    return blend.unsqueeze(0)

# Computes average CLIP image embeddings using Test-Time Augmentation (TTA) with original and horizontally flipped images.
def clip_image_feats_with_tta(pil_list, processor, model, return_flipped=False):
    """
    Compute averaged CLIP image features with simple TTA (original + horizontal flip).
    If return_flipped=True, also return the flipped image list.
    """
    device = next(model.parameters()).device
    flipped_imgs = [ImageOps.mirror(img) for img in pil_list]  # horizontal flips
    aug_batches = [pil_list, flipped_imgs]
    feats_accum = None

    for imgs in aug_batches:
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)   # (N,d)
        feats = F.normalize(feats, p=2, dim=-1)
        feats_accum = feats if feats_accum is None else feats_accum + feats

    feats_mean = F.normalize(feats_accum / len(aug_batches), p=2, dim=-1)

    if return_flipped:
        return feats_mean, flipped_imgs  # return both
    return feats_mean

def choose_image(target, sentence, images, image_dict, 
                 tokenizer=None, model=None, processor=None, blip_model=None, print_output=False):
    """Given a target word, sentence, and a list of candidate images, choose the image that best matches the target word

    Args:
        target (str): The target word
        sentence (str): The sentence
        images (list): List of the images
        image_dict (dict): Dictionary that maps image names to the images
        context_embedding (tensot): The contextual embedding of the target word in its given sentence
        tokenizer: The tokenizer
        model: The model
        processor: The BlipProcessor
        blip_model: The Blip model
        print_output: Whether or not to print the output out

    Returns:
        ranked_images (list): A list of the images ranked from highest similarity to lowest
        ranked_captions (list): The corresponding image captions
        ranked_embs (list): The corresponding caption embeddings
    """
    
    if isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel):
    
        # Get the contextual embedding for the target word in the short sentence
        if print_output: print("\nSentence:", sentence, "\nTarget:", target)

        # Find the definition that best fits the word        
        # Text embeddings (normalized)
        alphas = [0.6, 0.8]  # you can add more values if desired
        embeddings = []
        for a in alphas:
            emb = get_blended_text_embedding(target, sentence, processor, model, alpha=a)
            embeddings.append(emb)

        # 3. Average everything and normalize
        text_emb = F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=-1)  # (1, d)
        
        # Get embeddings for each of the images (batching, normalized)
        pil_batch = []
        valid_names = []
        for name in images:
            if name in image_dict:
                pil_batch.append(image_dict[name])
                valid_names.append(name)
            else:
                # Keep order consistent even if an image is missing
                pil_batch.append(Image.new('RGB', (224, 224)))
                valid_names.append(name)
        
        #Process in one batch
        with torch.no_grad():
            img_feats = clip_image_feats_with_tta(pil_batch, processor, model)  # (N,d)


        # # Get the cosine similarities (dot product after l2-normalization)
        sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()  # (N,)
        def sharp_prompts(target, sentence):
            return [
                f"the intended meaning of '{target}' in: {sentence}",
                f"correct interpretation of '{target}' here: {sentence}",
                f"what '{target}' means in context: {sentence}",
            ]
        
        # Get top-k indices from the first pass
        ranked_indices = np.argsort(sims)[::-1].copy()
        topk = ranked_indices[: min(5, len(ranked_indices))]  # top-3 for rerank

        # Compute sharper query embedding (averaged sharp prompts)
        sharp_q = get_clip_text_embedding_multi(sharp_prompts(target, sentence), processor, model).unsqueeze(0)
        sharp_sims = (sharp_q @ img_feats[topk].T).squeeze(0).detach().cpu().numpy()

        # Blend original and sharp similarities for top-k
        beta = 0.5  # weight between original (sims) and sharp (re-rank)
        sims[topk] = beta * sims[topk] + (1 - beta) * sharp_sims

        # Final ranking after re-blending
        ranked_indices = np.argsort(sims)[::-1]
        ranked_images = [valid_names[int(i)] for i in ranked_indices]

        ranked_captions = [None for _ in ranked_images]
        ranked_embs = [None for _ in ranked_images]

        if print_output:
            i = 0  # just the first image
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(pil_batch[i])
            axes[0].set_title(f"Original {valid_names[i]}")
            axes[0].axis("off")
            axes[1].set_title(f"Flipped (TTA) {valid_names[i]}")
            axes[1].axis("off")
            plt.tight_layout()
            plt.show(block=True)
            for rank, i in enumerate(ranked_indices):
                # image_name, caption, emb = image_embeddings[i]
                # clearplt.imshow(image_dict[valid_names[i]])
                # plt.title(f"Rank {rank+1}\nSentence: {sentence} --> Definition: {best_syn.definition()}\nSimilarity: {sims[i]}, Caption: {caption}")
                plt.title(f"Rank {rank+1} | sim={sims[i]:.4f}\nSentence: {sentence}")
                plt.axis('off')
                # plt.show()
            # print("Ranked Images:", ranked_images)
            # print("Ranked Captions:", ranked_captions)

        return ranked_images, ranked_captions, ranked_embs

if __name__ == "__main__":
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"

    file_path = "dataset"   # File path to the dataset (which should contain the folders test_v1, train_v1, and trial_v1)
    print_output = False    # Set to True to visualize the results

    # Define the various pretrained models
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPProcessor.from_pretrained(model_name)  # Tokenizer
    model = CLIPModel.from_pretrained(model_name).to(device)   # LLM which generates embeddings from text


    # Load in the data
    data, image_dict = load_data(file_path=file_path, train_val="trial")  # trial is for debugging (use train or test for evaluation)

    predicted_ranks = []
    for idx, row in data.iterrows():   # Iterate through the data
        target = row['target']
        sentence = row['sentence']
        images = [row[f'image_{i}'] for i in range(10)]  # Collect image filenames image_0 to image_9
        label = row['label']

        if label not in image_dict:
            print(f"Skipping {label} (not found in image_dict)")
            continue
        ranked_images, ranked_captions, ranked_embs = choose_image(
            target, sentence, images, image_dict,
            tokenizer=None, model=model, processor=tokenizer, blip_model=None, print_output=print_output
        )
        # TODO: Evaluate the model
        predicted_rank = ranked_images.index(label)+1  # Similarity rank of the image that should have been selected (lower is better)
        print("Predicted Rank:", predicted_rank)
        predicted_ranks.append(predicted_rank)

    predicted_ranks = np.array(predicted_ranks)
    mrr = np.mean(1/predicted_ranks)   # Mean reciprical rank
    hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)

    # Output the metrics
    print("---------------------------------")
    print(f"MRR: {mrr}")
    print(f"Hit Rate: {hit_rate}")
