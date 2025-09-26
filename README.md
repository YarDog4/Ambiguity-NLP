# Ambiguity-NLP:

This GitHub repository contains notebooks for our COSC 524 final project on Visual Word Sense Disambiguation (VWSD). As a group, we implement and compare different tokenization approaches (character-level, subword, and word embeddings) to establish text-only baselines. We then integrate image embeddings (via CLIP) to test how visual context shifts similarity scores and reduces linguistic ambiguity. Finally, we calculate cosine similarity scores of captions containing homographs (e.g., bank, jam, bat) to measure ambiguity.  The project explores how tokenization and embeddings interact with multimodal models, and quantifies the extent to which images improve disambiguation of ambiguous captions.

The above is subject to change.
Notebook hygiene
----------------
If you downloaded or edited a notebook in Google Colab, the repo provides a small cleaner script that will remove problematic top-level widget metadata and ensure minimal `nbformat` metadata. Running the cleaner is optional — CI also sanitizes notebooks on push/PR.

Quick manual cleaning (one-off):

1. From a terminal in the repository run the cleaner against a notebook you downloaded from Colab:

```bash
python3 tools/clean_notebooks.py path/to/Step_1_Ambiguous_Reasoning_1.ipynb
```

2. Inspect the notebook or open it in VS Code — the top-level `metadata.widgets` will be removed and minimal `nbformat` metadata added if needed.

If you'd like, I can add a tiny `scripts/clean_all_notebooks.sh` that runs the cleaner across the repo so manual cleaning is a single command for anyone who prefers that option.


Task 1: Tokenization & Embedding Pipeline (by: Danny & Shamik)

Task 2: Cosine Similarity Experiments with Homographs (we should all work on this together)

Task 3: Image-based Disambiguation with CLIP (or something more recent) - (we should all work on this together)

Task 4: Lexical + Visual Knowledge Integration (WordNet + Vision)? - (we should all work on this together)
