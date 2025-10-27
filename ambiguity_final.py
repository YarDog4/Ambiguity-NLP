import os
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
        # context_embedding = get_context(sentence, target, tokenizer=tokenizer, model=model)

        # Find the definition that best fits the word
        
        # Text embeddings (normalized)
        text_emb = get_sentence_embedding(sentence, tokenizer=processor, model=model)
        text_emb = text_emb.unsqueeze(0) #(1, d)
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
                pil_batch.append(Image.new('RGB', (224, 224)))
                valid_names.append(name)
        
        #Process in one batch
        with torch.no_grad():
            inputs = processor(images=pil_batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            img_feats = model.get_image_features(**inputs)   # (N, d)
            img_feats = F.normalize(img_feats, p=2, dim=-1)
    
        # # Get the cosine similarities (dot product after l2-normalization)
        sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()  # (N,)
        
        # Get indices of images sorted by similarity (highest first)
        ranked_indices = np.argsort(sims)[::-1]
        ranked_images = [valid_names[i] for i in ranked_indices]
       
        if print_output:
            for rank, i in enumerate(ranked_indices):
                plt.imshow(image_dict[valid_names[i]])
                plt.title(f"Rank {rank+1} | sim={sims[i]:.4f}\nSentence: {sentence}")
                plt.axis('off')
                plt.show()
            print("Ranked Images:", ranked_images)
            print("Ranked Captions:", ranked_captions)

        ranked_captions = [None for _ in ranked_images]
        ranked_embs = [None for _ in ranked_images]

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

        # processor=tokenizer (CLIPProcessor), model=model (CLIPModel); blip_model not needed
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

    print("---------------------------------")
    print(f"MRR: {mrr}")
    print(f"Hit Rate: {hit_rate}")

    # FUTURE WORK
    # TODO: For choosing the best definition
        # TODO: Rather than using all WordNet definitions, only use the ones for the correct part of speech
        # TODO: Use synonyms on the target word and get embeddings from them as well
        # TODO: Try changing the language of the word and choosing definitions there (as the paper does)
    # TODO: Vision Language Model to get text embeddings from images
        # TODO: Get a good dataset for images
        # TODO: Find best VLM (Clip, maybe something else)
            # TODO: The current code uses captions generated from Blip
    # TODO: For the final prediction
        # TODO: Should we use image captions with context, or embeddings without context?
        # TODO: Should we clean the captions to remove the given sentence?
    # TODO: Find a way to determine if we should use the definition embedding, the image embedding, or both to make the prediction
        # TODO: Maybe we can weight all of them based on ther similarity, using some sort of cross entropy with tempurature