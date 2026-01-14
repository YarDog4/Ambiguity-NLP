# only_augmentations.py
from core import setup_clip_and_nltk, load_data, evaluate_subset_configurable
from core import profile_inference_cost_for_config

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

    # fixed backbone + fixed temp
    TEMP = 0.7

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
