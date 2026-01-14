# both.py
from core import setup_clip_and_nltk, load_data, evaluate_subset_configurable
from core import profile_inference_cost_for_config

def main():
    file_path = "dataset"
    device, model, processor = setup_clip_and_nltk()

    test_df, test_images = load_data(file_path=file_path, train_val="test")

    TEMP = 0.7
    PHOTO_W = 0.6
    SEM_W = 0.4

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

    mrr, hit = evaluate_subset_configurable(
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

    print("\n=== BOTH (PROMPTING + AUGMENTATIONS) ===")
    print(f"Device: {device}")
    print(f"Temp: {TEMP} | photo_w: {PHOTO_W} | sem_w: {SEM_W}")
    print(f"MRR: {mrr:.6f}")
    print(f"Hit@1: {hit:.6f}")

if __name__ == "__main__":
    main()



# Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
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
#   Mean: 13.25 ms | P50: 11.99 | P90: 13.95 | P95: 20.71

# Image embedding latency:
#   Mean: 59.44 ms/image | P50: 60.58 | P90: 61.09 | P95: 61.50
#   Est. image-embedding per query (10 candidates): 594.41 ms/query

# End-to-end ranking latency (10 candidates):
#   Mean: 714.16 ms/query | P50: 716.84 | P90: 720.01 | P95: 720.71
# ==============================


# === BOTH (PROMPTING + AUGMENTATIONS) ===
# Device: mps
# Temp: 0.7 | photo_w: 0.6 | sem_w: 0.4
# MRR: 0.731490
# Hit@1: 0.585313