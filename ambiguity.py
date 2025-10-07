import os
import torch
import pandas as pd
import numpy as np
from PIL import Image

import nltk
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


############################## Load in the SemEval data ##############################

def load_data(file_path, train_val="train", target_size=(384, 384)):
        """Load in the data

        Args:
            file_path (str): The file path
            train_val (str): Whether to load in the train, test, or trial set
            target_size (tuple): The size of each image to use
                Use (224, 224) for CLIP only, (384, 384) for BLIP

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
            if filename.lower().endswith('.jpg'): 
                img = Image.open(os.path.join(path_images, filename)).convert('RGB')
                img_resized = img.resize(target_size, resample=Image.BICUBIC)
                image_dict[filename] = img_resized

        return data, image_dict


############################## Get Embeddings ##############################

def get_definition_embedding(text, tokenizer=None, model=None):   # Get an embedding for a definition or sentence
    """Get the sentence embedding by mean-pooling the last hidden states from BERT
    
    Args:
        text (str): The input text
        tokenizer: The tokenizer
        model: The model
        
    Returns:
        embedding (Tensor): The output embedding
    """
    if tokenizer is None: tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Initialize tokenizer and model
    if model is None: model = BertModel.from_pretrained('bert-base-uncased')

    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
        last_hidden = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
        embedding = last_hidden.mean(dim=1)[0]  # Mean-pool across all tokens (dim=1)
    return embedding

def get_context(sentence, target, tokenizer=None, model=None):  # Contextual Embedding with BERT
    """Get the contextual embedding for a target word, given the context

    Args:
        sentence (str): The sentence (must contain the target word)
        target (str): The targe word
        tokenizer: The tokenizer
        model: The model

    Returns:
        target_embedding (Tensor): The output embedding
    """
    if tokenizer is None: tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Initialize tokenizer and model
    if model is None: model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize sentence (with offsets)
    tokens = tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True)
    input_ids = tokens['input_ids']
    offsets = tokens['offset_mapping'][0]
    sentence_token_ids = input_ids[0].tolist()

    # Tokenize target (without special tokens)
    target_token_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    # Find the target's token indices in the sentence
    def find_sublist_index(big_list, sub_list):
        for i in range(len(big_list) - len(sub_list) + 1):
            if big_list[i:i+len(sub_list)] == sub_list:
                return list(range(i, i+len(sub_list)))
        return []

    target_indices = find_sublist_index(sentence_token_ids, target_token_ids)

    if not target_indices:
        print(f"Target word '{target}' not found in sentence.")
        target_embedding = None
    else:
        tokens_for_model = {k: v for k, v in tokens.items() if k != "offset_mapping"}  # Remove 'offset_mapping' before passing to model
        with torch.no_grad():
            outputs = model(**tokens_for_model)   # Get the embeddings by passing it into the neural network
            last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        target_embedding = last_hidden[0, target_indices, :].mean(dim=0)   # Mean pool embeddings across the matched subword token indices

    return target_embedding


############################## Choose the Best Definitional Embedding ##############################

def choose_definition(target, context_embedding, tokenizer=None, model=None, print_output=False):
    """Given a target word and sentence

    Args:
        target (str): The targe word
        context_embedding (tensot): The contextual embedding of the target word in its given sentence
        tokenizer: The tokenizer
        model: The model

    Returns:
        best_syn: The best sense of the word
            best_syn.name() is the name of the sense
            best_syn.definition() is the chosen definition
        best_emb: The chosen dictionary embedding
    """
    if tokenizer is None: tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Initialize tokenizer and model
    if model is None: model = BertModel.from_pretrained('bert-base-uncased')

    # Get the embeddings for each defintiion
    definition_embeddings = []
    for syn in wn.synsets(target): 
        definition = syn.definition()
        # print(syn.name(), ":", definition)
        definition_embedding = get_definition_embedding(definition)  # Get the embedding for the definition
        definition_embeddings.append((syn, definition_embedding))   # Store the sense and definition embedding

    # Convert to numpy arrays for similarity
    context_vec = context_embedding.detach().cpu().numpy().reshape(1, -1)
    definition_vecs = [emb.detach().cpu().numpy().reshape(1, -1) for _, emb in definition_embeddings]
    definition_vecs_np = np.vstack(definition_vecs)

    sims = cosine_similarity(context_vec, definition_vecs_np)[0]
    best_idx = sims.argmax()
    best_syn, best_emb = definition_embeddings[best_idx]

    print("The cosine similarities for each definition are:")
    if print_output:  # Print cosine similarity, sense, definition
        for i, syn in enumerate(wn.synsets(target)): 
            print(sims[i], syn.name(), ":", syn.definition())
        print("Best sense:", best_syn.name())
        print("Definition:", best_syn.definition())

    return best_syn, best_emb


if __name__ == "__main__":

    # Define our tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Initialize tokenizer and model
    model = BertModel.from_pretrained('bert-base-uncased')

    # Download wordnet definitions
    nltk.download('wordnet')   # Download WordNet
    nltk.download('omw-1.4')   # Download multilingual WordNet
    print()

    # Load in the data
    file_path = "dataset"   # File path to the dataset (which should contain the folders test_v1, train_v1, and trial_v1)
    data, image_dict = load_data(file_path=file_path, train_val="trial")

    for idx, row in data.iterrows():   # Iterate through the data
        target = row['target']
        sentence = row['sentence']
        images = [row[f'image_{i}'] for i in range(10)]  # Collect image filenames image_0 to image_9
        label = row['label']

        # Get the contextual embedding for the target word in the short sentence
        print("\nSentence:", sentence, "\nTarget:", target)
        context_embedding = get_context(sentence, target, tokenizer=tokenizer, model=model)
        best_syn, best_emb = choose_definition(target, context_embedding, tokenizer=tokenizer, model=model, print_output=True)

    # TODO: For choosing the best definition
        # TODO: Rather than using all WordNet definitions, only use the ones for the correct part of speech
        # TODO: Use synonyms on the target word and get embeddings from them as well
        # TODO: Try changing the language of the word and choosing definitions there (as the paper does)
    # TODO: Vision Language Model to get text embeddings from images
        # TODO: Get a good dataset for images
        # TODO: Find best VLM (Clip, maybe something else)
    # TODO: Find a way to determine if we should use the definition embedding, the image embedding, or both to make the prediction
        # TODO: Maybe we can weight all of them based on ther similarity, using some sort of cross entropy with tempurature
    # TODO: Actually make predictions
    # TODO: Get evaluation metrics to compare to the paper