# Partial Evaluation Metrics for Named Entity Recognition

**Authors:** Charles Sullivan and James Kong

## Implemented Evaluation Metrics

- Exact: exact type and boundary match
- Left boundary and type match
- Right boundary match and type match
- Partial boundary match: any overlap in predicted and reference spans, type matches
    - A gold mention can have no more than 1 matching partial prediction
    - A prediction cannot partially match more than 1 gold mention  

## Motivation

Named Entity Recognition (NER) evaluation currently sits between two extremes:
- **Token-level F1**: Too lenient - identifying just one token (e.g., "New" in "New York") doesn't capture the complete entity.
- **Exact phrase-level F1**: Too strict - requires perfect boundary and type matching, creating both false positives and negatives for minor boundary differences.

Consider these examples:
- Gold: "The Ohio State University"[ORG] is in "Ohio"[LOC].
- Prediction: The "Ohio State University"[ORG] is in "Ohio"[LOC].

Or:
- Gold: "The New York Times"[ORG]
- Prediction: The "New York Times"[ORG]

These predictions capture the correct entities but are penalized as errors under exact matching. We believe partial credit would better reflect model performance.

## Project Overview

This project explores evaluation metrics that award partial credit to NER predictions that are close but not exact matches to annotated data. We will:

1. Implement various partial evaluation metrics (based on SemEval 2013 and other approaches)
2. Train NER models using FLAIR with CRF on Twitter data
3. Evaluate model outputs using different metrics
4. Conduct human error analysis to determine how well metrics align with human judgment
5. Compare partial metrics against baseline token-level and exact-match F1

## Dataset

We will work primarily with Twitter data, where exact matching is particularly challenging:
- [Broad Twitter Corpus](https://aclanthology.org/C16-1111.pdf)

## Methodology

### Model Implementation
- Using [FLAIR](https://github.com/flairNLP/flair) with CRF to minimize invalid label sequences
- Focus on comparing evaluation metrics rather than optimizing model architecture

### Partial Evaluation Metrics
We will implement and analyze several metrics:
- Left match: Left boundary + category matches (good for relation extraction)
- Right match: Useful for missing/included preceding adjectives
- Partial match: Any fragment overlap
- Approximate match: Substring of annotation
- Fragment percentage: What percentage of the entity is correctly identified
- Categorical relaxation: Relaxing ontology constraints
- Scoring weights: Award different partial credits (e.g., 0.5) for partially correct annotations

### Analysis
- Quantitative comparison of how different metrics score the same predictions
- Tune models using each scoring metric and compare the results
- Human evaluation to determine which metrics best align with human judgment
- Error analysis to identify patterns in partial matches

## References

- [SemEval 2013 Task 9.1](https://aclanthology.org/S13-2056.pdf)
- [Named Entity Recognition Evaluation](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)
- [Biomedical NER](https://link.springer.com/article/10.1186/1471-2105-7-92)
- [FLAIR Documentation](https://flairnlp.github.io/docs/tutorial-training/how-to-train-sequence-tagger#training-a-named-entity-recognition-ner-model-with-flair-embeddings)