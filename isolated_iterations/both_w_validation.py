# both_w_validation_split.py
import numpy as np
from core import setup_clip_and_nltk, load_data, evaluate_subset_configurable
from core import profile_inference_cost_for_config

def main():
    file_path = "dataset"
    device, model, processor = setup_clip_and_nltk()

    train_df, train_images = load_data(file_path=file_path, train_val="train")
    test_df, test_images   = load_data(file_path=file_path, train_val="test")

    # fixed settings (no Optuna)
    TEMP = 0.7
    PHOTO_W = 0.6
    SEM_W = 0.4

    # train/val split (only for reporting / sanity check)
    np.random.seed(42)
    idx = np.arange(len(train_df))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    val_df = train_df.iloc[va_idx].reset_index(drop=True)

    # profile validation cost
    profile_inference_cost_for_config(
        df=val_df,
        image_dict=train_images,
        model=model,
        processor=processor,
        use_prompting=True,
        use_augmentations=True,
        photo_weight=PHOTO_W,
        semantic_weight=SEM_W,
        temp=TEMP,
        num_rows=50,
    )

    # profile test cost
    profile_inference_cost_for_config(
        df=test_df,
        image_dict=test_images,
        model=model,
        processor=processor,
        use_prompting=True,
        use_augmentations=True,
        photo_weight=PHOTO_W,
        semantic_weight=SEM_W,
        temp=TEMP,
        num_rows=50,
    )

    val_mrr, val_hit = evaluate_subset_configurable(
        val_df,
        image_dict=train_images,
        model=model,
        processor=processor,
        use_prompting=True,
        use_augmentations=True,
        photo_weight=PHOTO_W,
        semantic_weight=SEM_W,
        temp=TEMP,
    )

    test_mrr, test_hit = evaluate_subset_configurable(
        test_df,
        image_dict=test_images,
        model=model,
        processor=processor,
        use_prompting=True,
        use_augmentations=True,
        photo_weight=PHOTO_W,
        semantic_weight=SEM_W,
        temp=TEMP,
    )

    print("\n=== BOTH + VALIDATION SPLIT (NO BAYES) ===")
    print(f"Device: {device}")
    print(f"Temp: {TEMP} | photo_w: {PHOTO_W} | sem_w: {SEM_W}")
    print(f"VAL  MRR: {val_mrr:.6f} | VAL  Hit@1: {val_hit:.6f}")
    print(f"TEST MRR: {test_mrr:.6f} | TEST Hit@1: {test_hit:.6f}")

if __name__ == "__main__":
    main()



# Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
# Loading cached images from dataset/train_v1/image_cache.pkl...
# Loaded 12995 cached images
# Loading cached images from dataset/test_v1/image_cache.pkl...
# Loaded 8081 cached images

# ==============================
# COMPUTATIONAL COST PROFILE
# ==============================
# Config: prompting=True | augmentations=True
# Device: mps:0
# Profiled queries: 50
# Candidates/query: 10
# Views per image: 28

# Text embedding latency:
#   Mean: 22.46 ms | P50: 12.13 | P90: 33.28 | P95: 37.81

# Image embedding latency:
#   Mean: 59.60 ms/image | P50: 60.68 | P90: 63.09 | P95: 63.95
#   Est. image-embedding per query (10 candidates): 596.00 ms/query

# End-to-end ranking latency (10 candidates):
#   Mean: 716.43 ms/query | P50: 715.43 | P90: 728.17 | P95: 733.22
# ==============================


# ==============================
# COMPUTATIONAL COST PROFILE
# ==============================
# Config: prompting=True | augmentations=True
# Device: mps:0
# Profiled queries: 50
# Candidates/query: 10
# Views per image: 28

# Text embedding latency:
#   Mean: 13.40 ms | P50: 13.18 | P90: 14.65 | P95: 14.79

# Image embedding latency:
#   Mean: 59.09 ms/image | P50: 60.36 | P90: 60.86 | P95: 61.13
#   Est. image-embedding per query (10 candidates): 590.93 ms/query

# End-to-end ranking latency (10 candidates):
#   Mean: 712.32 ms/query | P50: 715.18 | P90: 717.51 | P95: 718.68
# ==============================


# === BOTH + VALIDATION SPLIT (NO BAYES) ===
# Device: mps
# Temp: 0.7 | photo_w: 0.6 | sem_w: 0.4
# VAL  MRR: 0.851082 | VAL  Hit@1: 0.770785
# TEST MRR: 0.727206 | TEST Hit@1: 0.576674