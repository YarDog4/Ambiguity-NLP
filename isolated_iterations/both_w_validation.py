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

    # train/val split (only for reporting / sanity check)
    np.random.seed(42)
    idx = np.arange(len(train_df))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    val_df = train_df.iloc[va_idx].reset_index(drop=True)

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
