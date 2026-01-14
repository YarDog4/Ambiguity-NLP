# only_augmentations.py
from core import setup_clip_and_nltk, load_data, evaluate_subset_configurable
from core import profile_inference_cost_for_config

# fixed backbone + fixed temp
TEMP = 0.7

def main():
    file_path = "dataset"
    device, model, processor = setup_clip_and_nltk()

    # evaluate on test
    test_df, test_images = load_data(file_path=file_path, train_val="test")

    profile_inference_cost_for_config(
        df=test_df,
        image_dict=test_images,
        model=model,
        processor=processor,
        use_prompting=False,
        use_augmentations=True,
        temp=TEMP,
        num_rows=50,
    )

    mrr, hit = evaluate_subset_configurable(
        test_df,
        image_dict=test_images,
        model=model,
        processor=processor,
        use_prompting=False,
        use_augmentations=True,
        temp=TEMP,
    )

    print("\n=== ONLY AUGMENTATIONS ===")
    print(f"Device: {device}")
    print(f"Temp: {TEMP}")
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
# Config: prompting=False | augmentations=True
# Device: mps:0
# Profiled queries: 50
# Candidates/query: 10
# Views per image: 28

# Text embedding latency:
#   Mean: 23.68 ms | P50: 5.34 | P90: 6.51 | P95: 147.08

# Image embedding latency:
#   Mean: 59.65 ms/image | P50: 60.71 | P90: 62.47 | P95: 63.59
#   Est. image-embedding per query (10 candidates): 596.47 ms/query

# End-to-end ranking latency (10 candidates):
#   Mean: 709.77 ms/query | P50: 710.21 | P90: 722.17 | P95: 729.41
# ==============================


# === ONLY AUGMENTATIONS ===
# Device: mps
# Temp: 0.7
# MRR: 0.722540
# Hit@1: 0.574514