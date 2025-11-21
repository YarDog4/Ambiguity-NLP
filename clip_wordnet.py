import os
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import spacy
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F

import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity



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


############################## Get Embeddings ##############################


def get_sentence_embedding(text, tokenizer=None, model=None):
    """Get the sentence embedding.

    If (tokenizer is CLIPProcessor) and (model is CLIPModel), return a normalized
    CLIP text embedding in the image-text space.

    Otherwise, fall back to a generic transformer encoder that returns last_hidden_state
    and mean-pools across tokens.

    Args:
        text (str): The input text
        tokenizer: The tokenizer
        model: The model

    Returns:
        embedding (Tensor): The output embedding of shape (hidden_dim,)
    """
    # CLIP branch
    if isinstance(tokenizer, CLIPProcessor) and isinstance(model, CLIPModel):
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = tokenizer(
                text=[text], return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_text_features(**inputs)  # (1, d)
            features = F.normalize(features, p=2, dim=-1)
        return features[0]

    # Generic transformer branch (if you ever pass a non-CLIP model)
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        last_hidden = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
        embedding = last_hidden.mean(dim=1)[0]  # Mean-pool across all tokens (dim=1)
    return embedding


############################## Choose the Best Definitional Embedding ##############################


def choose_definition(target, sentence, tokenizer=None, model=None, print_output=False, ner=None, filter_for_pos=True):
    """
    Given a target word and its context sentence, choose the WordNet definition
    whose CLIP text embedding is most similar to the sentence's CLIP embedding.

    Args:
        target (str): The target word
        sentence (str): The full sentence containing the target
        tokenizer: The tokenizer (e.g. CLIPProcessor)
        model: The text model (e.g. CLIPModel)
        print_output (bool): Whether to print similarities and chosen sense
        ner (spacy model): Named Entity Recognition model
        filter_for_pos (bool): Wether or not to filter for the POS

    Returns:
        best_syn: The best WordNet synset (or None if none exist)
        best_emb (Tensor): The CLIP text embedding of the best definition
                           (or the sentence embedding if no synsets found)
        context_embedding (Tensor): The CLIP text embedding of the whole sentence
    """

    # Load the pretrained English model
    if ner is None: ner = spacy.load('en_core_web_sm')

    if tokenizer is None or model is None:
        raise ValueError(
            "Tokenizer and model (e.g. CLIPProcessor and CLIPModel) must be provided."
        )

    # Sanitize inputs: ensure strings and handle NaN/None
    if target is None or (isinstance(target, float) and np.isnan(target)):
        # No usable target; fall back to sentence embedding
        target = ""
    if not isinstance(target, str):
        target = str(target)
    if sentence is None or (isinstance(sentence, float) and np.isnan(sentence)):
        sentence = ""
    if not isinstance(sentence, str):
        sentence = str(sentence)

    # Use the full sentence as context representation
    context_text = sentence
    context_embedding = get_sentence_embedding(
        context_text, tokenizer=tokenizer, model=model
    )
    context_embedding = F.normalize(context_embedding, p=2, dim=-1)  # (d,)

    synsets = wn.synsets(target)
    if not synsets:
        # No WordNet entries, fall back to the sentence embedding
        if print_output:
            print(
                f"No synsets found for '{target}', falling back to sentence embedding."
            )
        return None, context_embedding, context_embedding

    definition_embeddings = []
    # Process the sentence
    doc = ner(sentence)
    for token in doc:  # Find target token and get its POS tag
        if token.text == target:
            print(f"Target word: {token.text}, POS tag: {token.pos_}")
            pos = token.pos_
            break
    wordnet_pos = spacy_to_wordnet_pos(pos)

    if filter_for_pos: filtered_synsets = [syn for syn in synsets if syn.pos() == wordnet_pos]
    else: filtered_synsets = synsets
    if len(filtered_synsets)==0: filtered_synsets = synsets
    for syn in filtered_synsets:
        definition = syn.definition()
        definition_embedding = get_sentence_embedding(
            definition, tokenizer=tokenizer, model=model
        )
        definition_embedding = F.normalize(definition_embedding, p=2, dim=-1)  # (d,)
        definition_embeddings.append((syn, definition_embedding))

    # Stack definition embeddings and compute cosine similarity with context
    def_matrix = torch.stack([emb for _, emb in definition_embeddings], dim=0)  # (N, d)
    # embeddings are normalized -> dot product = cosine similarity
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


############################## Connect Images To Text ##############################


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
    embedding_weights=[0.5,0.5],
    print_output=False,
):
    """Given a target word, sentence, and a list of candidate images, choose the image that best matches the target word

    Args:
        target (str): The target word
        sentence (str): The sentence
        images (list): List of the images
        image_dict (dict): Dictionary that maps image names to the images
        tokenizer: (unused for CLIP branch; kept for backward-compat)
        model: The model (CLIPModel)
        processor: The CLIPProcessor
        blip_model: (unused here)
        ner (Spacy model): Named entity recofnitin model
        filter_for_pos (bool): Wether or not to filter for the POS
        print_output: Whether or not to print the output out

    Returns:
        ranked_images (list): A list of the images ranked from highest similarity to lowest
        ranked_captions (list): The corresponding image captions (None here)
        ranked_embs (list): The corresponding caption embeddings (None here)
    """
    if ner is None: ner = spacy.load('en_core_web_sm')

    # CLIP branch
    if isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel):

        if print_output:
            print("\nSentence:", sentence, "\nTarget:", target)

        # 1) Use WordNet + CLIP to pick the best sense
        #    tokenizer=processor (CLIPProcessor), model=model (CLIPModel)
        best_syn, best_definition_emb, context_embedding = choose_definition(
            target,
            sentence,
            tokenizer=processor,
            model=model,
            print_output=print_output,
            ner=ner,
            filter_for_pos=filter_for_pos
        )

        # 2) Use the best definition embedding as the text query in CLIP space.
        #    If WordNet had no synsets, best_definition_emb is just the sentence embedding.
        mean_embedding = embedding_weights[0]*best_definition_emb + embedding_weights[1]*context_embedding
        text_emb = mean_embedding.unsqueeze(0)  # (1, d)
        device = next(model.parameters()).device


        # Get embeddings for each of the images (batching, normalized)
        pil_batch = []
        valid_names = []
        for name in images:
            if name in image_dict:
                pil_batch.append(image_dict[name])
                valid_names.append(name)
            else:
                # Keep order consistent even if an image is missing
                pil_batch.append(Image.new("RGB", (224, 224)))
                valid_names.append(name)

        # Process in one batch
        with torch.no_grad():
            inputs = processor(images=pil_batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            img_feats = model.get_image_features(**inputs)  # (N, d)
            img_feats = F.normalize(img_feats, p=2, dim=-1)

        # Cosine similarities via dot product (because embeddings normalized)
        sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()  # (N,)

        # Get indices of images sorted by similarity (highest first)
        ranked_indices = np.argsort(sims)[::-1]
        ranked_images = [valid_names[i] for i in ranked_indices]

        # if print_output:
        #     for rank, i in enumerate(ranked_indices):
        #         plt.imshow(pil_batch[i])
        #         title = f"Rank {rank+1} | sim={sims[i]:.4f}\nSentence: {sentence}"
        #         if best_syn is not None:
        #             title += (
        #                 f"\nSense: {best_syn.name()} | Def: {best_syn.definition()}"
        #             )
        #         plt.title(title)
        #         plt.axis("off")
        #         plt.show()
        #     print("Ranked Images:", ranked_images)

        ranked_captions = [None for _ in ranked_images]
        ranked_embs = [None for _ in ranked_images]

        return ranked_images, ranked_captions, ranked_embs

    # If not CLIP, you could add a fallback here if needed
    raise ValueError("Processor and model must be CLIPProcessor and CLIPModel.")


############################## Main ##############################


if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    file_path = (
        "dataset"  # File path to the dataset (which should contain the folders test_v1,
        # train_v1, and trial_v1)
    )
    print_output = True  # Set to True to visualize the results

    # Download WordNet data if needed
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # Define the pretrained CLIP model
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)  # Processor
    model = CLIPModel.from_pretrained(model_name).to(device)  # CLIP model
    ner = spacy.load('en_core_web_sm')
    

    # Load in the data
    data, image_dict = load_data(
        file_path=file_path, train_val="trial"
    )  # trial is for debugging (use train or test for evaluation)

    predicted_ranks = []
    for idx, row in data.iterrows():  # Iterate through the data
        target = row["target"]
        sentence = row["sentence"]
        images = [row[f"image_{i}"] for i in range(10)]  # image_0 to image_9
        label = row["label"]

        ranked_images, ranked_captions, ranked_embs = choose_image(
            target,
            sentence,
            images,
            image_dict,
            tokenizer=None,  # unused in CLIP branch
            model=model,
            processor=processor,
            blip_model=None,
            ner=ner,
            filter_for_pos=False,    # Whether or not to filter for the POS
            embedding_weights=[0.15,0.85],   # How to weight definition embedding and sentence embedding
            print_output=print_output,
        )

        # Evaluate the model: rank of the correct image (lower is better)
        predicted_rank = ranked_images.index(label) + 1
        print("Predicted Rank:", predicted_rank)
        predicted_ranks.append(predicted_rank)

    predicted_ranks = np.array(predicted_ranks)
    mrr = np.mean(1 / predicted_ranks)  # Mean reciprocal rank
    hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)

    print("---------------------------------")
    print(f"MRR: {mrr}")
    print(f"Hit Rate: {hit_rate}")


