import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F


############################## Load in the SemEval data ##############################

def load_data(file_path, train_val="train", target_size=(384, 384)):
    """Load the dataset and associated images."""
    path = os.path.join(file_path, train_val + "_v1")

    # Load tab-separated text data (target, sentence, 10 image names)
    path_data = os.path.join(path, train_val + ".data.v1.txt")
    data = pd.read_csv(path_data, sep='\t', header=None)
    data.columns = ['target', 'sentence'] + [f'image_{i}' for i in range(data.shape[1] - 2)]

    # Load the correct (gold) labels
    path_labels = os.path.join(path, train_val + ".gold.v1.txt")
    with open(path_labels, "r") as f:
        gold_labels = [line.strip() for line in f]
    data['label'] = gold_labels

    # Load images into a dictionary
    path_images = os.path.join(path, train_val + "_images_v1")
    image_dict = {}
    for filename in tqdm(os.listdir(path_images), desc="Loading Images", unit="image"):
        if filename.lower().endswith(('.jpg', '.png')):
            try:
                img = Image.open(os.path.join(path_images, filename)).convert('RGB')
                image_dict[filename] = img
            except Exception:
                continue

    return data, image_dict


############################## TTA (Flip Augmentation) ##############################

def clip_image_feats_with_tta(pil_list, processor, model, return_flipped=True):
    """
    Compute averaged CLIP image features using Test-Time Augmentation (TTA):
    - original image
    - horizontally flipped image
    """
    device = next(model.parameters()).device
    flipped_imgs = [ImageOps.mirror(img) for img in pil_list]  # horizontal flips
    aug_batches = [pil_list, flipped_imgs]
    feats_accum = None

    for imgs in aug_batches:
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)   # (N, d)
        feats = F.normalize(feats, p=2, dim=-1)
        feats_accum = feats if feats_accum is None else feats_accum + feats

    feats_mean = F.normalize(feats_accum / len(aug_batches), p=2, dim=-1)

    if return_flipped:
        return feats_mean, flipped_imgs
    return feats_mean


############################## Sentence Embedding ##############################

def get_sentence_embedding(text, tokenizer=None, model=None):
    """Get CLIP text embedding normalized in image-text space."""
    if isinstance(tokenizer, CLIPProcessor) and isinstance(model, CLIPModel):
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = tokenizer(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_text_features(**inputs)
            features = F.normalize(features, p=2, dim=-1)
        return features[0]
    else:
        raise ValueError("Tokenizer and model must be CLIPProcessor and CLIPModel.")


############################## Main Similarity + Visualization ##############################

def choose_image(target, sentence, images, image_dict,
                 tokenizer=None, model=None, processor=None, print_output=True):
    """Choose the image that best matches the target word based on CLIP similarity."""

    if not (isinstance(processor, CLIPProcessor) and isinstance(model, CLIPModel)):
        raise ValueError("Processor and model must be CLIP components.")

    if print_output:
        print(f"\nSentence: {sentence}\nTarget: {target}")

    # --- 1. Get text embedding ---
    text_emb = get_sentence_embedding(sentence, tokenizer=processor, model=model).unsqueeze(0)

    device = next(model.parameters()).device

    # --- 2. Prepare image batch ---
    pil_batch, valid_names = [], []
    for name in images:
        if name in image_dict:
            pil_batch.append(image_dict[name])
            valid_names.append(name)
        else:
            pil_batch.append(Image.new('RGB', (224, 224)))  # dummy image
            valid_names.append(name)

    # --- 3. Compute image embeddings using flip-TTA ---
    with torch.no_grad():
        img_feats, flipped_imgs = clip_image_feats_with_tta(pil_batch, processor, model, return_flipped=True)

    # --- 4. Visualize the first original+flipped pair ---
    if print_output and len(pil_batch) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pil_batch[0])
        axes[0].set_title(f"Original: {valid_names[0]}")
        axes[0].axis("off")
        axes[1].imshow(flipped_imgs[0])
        axes[1].set_title(f"Flipped (TTA): {valid_names[0]}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show(block=True)
        input("Press Enter to close the image...")

    # --- 5. Compute similarities ---
    sims = (text_emb @ img_feats.T).squeeze(0).detach().cpu().numpy()
    ranked_indices = np.argsort(sims)[::-1]
    ranked_images = [valid_names[i] for i in ranked_indices]

    if print_output:
        print("\nTop Ranked Images:")
        for rank, i in enumerate(ranked_indices[:3]):  # show top 3 ranked
            print(f"Rank {rank+1}: {valid_names[i]} (sim={sims[i]:.4f})")

    ranked_captions = [None for _ in ranked_images]
    ranked_embs = [None for _ in ranked_images]
    return ranked_images, ranked_captions, ranked_embs


############################## Main Execution ##############################

if __name__ == "__main__":
    # --- 1. Select device ---
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"

    # --- 2. Load CLIP model ---
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    # --- 3. Load dataset ---
    file_path = "dataset"
    data, image_dict = load_data(file_path=file_path, train_val="trial")

    # --- 4. Evaluate dataset ---
    predicted_ranks = []
    for _, row in data.iterrows():
        target = row["target"]
        sentence = row["sentence"]
        images = [row[f"image_{i}"] for i in range(10)]
        label = row["label"]

        ranked_images, _, _ = choose_image(
            target, sentence, images, image_dict,
            model=model, processor=processor, print_output=False  # Set True to visualize one pair
        )

        predicted_rank = ranked_images.index(label) + 1
        print(f"Predicted Rank: {predicted_rank}")
        predicted_ranks.append(predicted_rank)

    # --- 5. Compute metrics ---
    predicted_ranks = np.array(predicted_ranks)
    mrr = np.mean(1 / predicted_ranks)
    hit_rate = np.sum(predicted_ranks == 1) / len(predicted_ranks)

    print("\n---------------------------------")
    print(f"MRR: {mrr:.4f}")
    print(f"Hit Rate: {hit_rate:.4f}")
