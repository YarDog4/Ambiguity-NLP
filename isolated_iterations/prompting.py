# only_prompting.py
from core import setup_clip_and_nltk, load_data, evaluate_subset_configurable
from core import profile_inference_cost_for_config

def main():
    file_path = "dataset"
    device, model, processor = setup_clip_and_nltk()

    test_df, test_images = load_data(file_path=file_path, train_val="trial")

    TEMP = 0.7
    PHOTO_W = 0.6
    SEM_W = 0.4

    profile_inference_cost_for_config(
        df=test_df,
        image_dict=test_images,
        model=model,
        processor=processor,
        use_prompting=True,
        use_augmentations=False,
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
        use_augmentations=False,
        photo_weight=PHOTO_W,
        semantic_weight=SEM_W,
        temp=TEMP,
    )

    print("\n=== ONLY PROMPTING ===")
    print(f"Device: {device}")
    print(f"Temp: {TEMP} | photo_w: {PHOTO_W} | sem_w: {SEM_W}")
    print(f"MRR: {mrr:.6f}")
    print(f"Hit@1: {hit:.6f}")

if __name__ == "__main__":
    main()



# Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
# Loading cached images from dataset/trial_v1/image_cache.pkl...
# Loaded 156 cached images

# ==============================
# COMPUTATIONAL COST PROFILE
# ==============================
# Config: prompting=True | augmentations=False
# Device: mps:0
# Profiled queries: 16
# Candidates/query: 10
# Views per image: 1

# Text embedding latency:
#   Mean: 39.43 ms | P50: 9.84 | P90: 85.22 | P95: 145.39

# Image embedding latency:
#   Mean: 5.28 ms/image | P50: 5.14 | P90: 5.80 | P95: 5.95
#   Est. image-embedding per query (10 candidates): 52.81 ms/query

# End-to-end ranking latency (10 candidates):
#   Mean: 62.95 ms/query | P50: 63.53 | P90: 67.99 | P95: 68.70
# ==============================


# === ONLY PROMPTING ===
# Device: mps
# Temp: 0.7 | photo_w: 0.6 | sem_w: 0.4
# MRR: 0.750595
# Hit@1: 0.625000