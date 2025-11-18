# Ambiguity-NLP:

This GitHub repository contains notebooks for our COSC 524 final project on Visual Word Sense Disambiguation (VWSD). As a group, we implement and compare different tokenization approaches (character-level, subword, and word embeddings) to establish text-only baselines. We then integrate image embeddings (via CLIP) to test how visual context shifts similarity scores and reduces linguistic ambiguity. Finally, we calculate cosine similarity scores of captions containing homographs (e.g., bank, jam, bat) to measure ambiguity.  The project explores how tokenization and embeddings interact with multimodal models, and quantifies the extent to which images improve disambiguation of ambiguous captions.

The above is subject to change.

Task 1: Tokenization & Embedding Pipeline (by: Danny & Shamik)

Task 2: Cosine Similarity Experiments with Homographs (we should all work on this together)

Task 3: Image-based Disambiguation with CLIP (or something more recent) - (we should all work on this together)

Task 4: Lexical + Visual Knowledge Integration (WordNet + Vision)? - (we should all work on this together)

Question: Since we’re using embeddings (Sentence-BERT, CLIP, etc.), our project is really encoder-focused in terms of Transformers. Do we want to explicitly say that in the README, or just leave it implied?


# How to run the code:

On a window's system, you can run this code by writing ```python filename.py``` in the terminal's command line.

The code that you should run to see progressive results include ```ambiguity.py```, ```ambiguity_final.py```, ```test.py```


## RESULTS:

### Trial Sets
Base Clip Approach:

`MRR: 0.734375`\
`Hit Rate: 0.5625`

Clip with Wordnet:

`MRR: 0.5355902777777778`\
`Hit Rate: 0.3125`

Translation Results Comparison
| Sentence Translation Experiment| MRR    | Hit Rate |
|--------------------------------|--------|----------|
| Baseline (English only)        | 0.7217 | 0.5788   |
| Spanish                        | 0.6989 | 0.5508   |
| French                         | 0.6958 | 0.5443   |  
| German                         | 0.6729 | 0.5162   |   
| Multi (ES+FR+DE)               | 0.6756 | 0.5313   |   
| Multi (ES+FR)                  | 0.6827 | 0.5292   |

