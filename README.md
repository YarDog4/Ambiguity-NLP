# Abstract:

Ambiguity poses persistent challenges in natural language understanding for both humans and Large Language Models (LLMs). Although LLMs excel at leveraging textual context to infer word meaning, they struggle when a word or phrase may hold multiple interpretations. Ambiguity exists in many forms such as lexical, syntactic, semantic, pragmatic, and more. We focus on lexical ambiguity, where one word has multiple distinct meanings. Specifically, we develop a model for Visual Word Sense Disambiguation (VWSD) on the SemEval-2023 dataset to predict which image best represents a target word in an ambiguous phrase. We use CLIP to generate textual and visual embeddings and cosine similarity to rank the candidate images. Expanding on the ideas of previous work, we refine our model by leveraging synonyms from WordNet, incorporating textual prompts, translating the sentences, and augmenting the images. This leads to a more generalizable model that can effectively resolve lexical ambiguity in settings with visual input.

# How to run the code:

On a window's system, you can run this code by running ```python final.py```. final.py is the file with the most recent model. All the other files are various scripts we used to debug and test earlier versions of the model.

The versions of Python used to run include Python 3.13.3 and 3.9.6.

The requirement.txt file includes the versions of the packages we used.


## RESULTS:

The dataset can be found at https://raganato.github.io/vwsd/

### Test Sets
Base Clip Approach:

`MRR: 0.7511`\
`Hit Rate: 0.6199`

