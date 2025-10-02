import torch
import numpy as np

import nltk
from transformers import BertTokenizerFast, BertModel
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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

    # Example sentence and target from the dataset
    target = "router"
    sentence = "internet router"

    # Define our tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Initialize tokenizer and model
    model = BertModel.from_pretrained('bert-base-uncased')

    # Download wordnet definitions
    nltk.download('wordnet')   # Download WordNet
    nltk.download('omw-1.4')   # Download multilingual WordNet
    print()

    # Get the contextual embedding for the target word in the short sentence
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
    # TODO: Integrate this script the the ambiguity dataset https://raganato.github.io/vwsd/
        # TODO: Actually make predictions
        # TODO: Get evaluation metrics to compare to the paper