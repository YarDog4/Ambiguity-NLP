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
