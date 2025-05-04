import os

from train import load_corpus
from flair.models import SequenceTagger
from flair.datasets import DataLoader, FlairDatapointDataset
from scorer import Scorer
from flair.data import Sentence
from tqdm import tqdm

from collections import Counter

def predict(data_points, model, batch_size, force_token_labels=False):

    # this is based on Flair's prediction method for their sequence tagger
    sentences = [sentence for sentence in data_points]

    dataloader = DataLoader(
        dataset=FlairDatapointDataset(sentences),
        batch_size=batch_size
    )

    print("Predicting...")
    for batch in dataloader:
        for sentence in batch:
            sentence.remove_labels('ner') # remove existing labels
        model.predict(batch, force_token_predictions=force_token_labels) # predict on the batch of sentences
    
    return sentences


def scorer_evaluate(reference, predictions):
    print("Evaluating...")
    scores = Scorer([], [])
    for reference, prediction in zip(reference, predictions):
        scores.merge(
            Scorer(
                Scorer.create_mentions(reference.get_labels()), 
                Scorer.create_mentions(prediction.get_labels())
                )
            )
    return scores


if __name__ == "__main__":
    # load corpus
    corpus = load_corpus()

    # extract the labels from the corpus
    # TODO why is add_unk=True required?
    label_type = 'ner'
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=True)

    # load model from file
    model_path = os.path.expanduser('~/.flair/models/best-model.pt')  # Ensure correct path
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = SequenceTagger.load(model_path)

    # evaluate with our Scorer
    predictions = predict(data_points=corpus.dev, model=model, batch_size=32)   

    # re-tagging the sentences means we lose the original dev tags
    # for now just reload the corpus, relatively fast
    corpus = load_corpus()
    scores = scorer_evaluate(corpus.dev, predictions)
    scores.print_score_report()

    # Write partial match CSVs
    scores.write_partial_matches("predictions/partial_dev_overlap.csv", match_type="overlap")
    scores.write_partial_matches("predictions/partial_dev_left_bound.csv", match_type="left")
    scores.write_partial_matches("predictions/partial_dev_right_bound.csv", match_type="right")

    # evaluate with Flair
    # do this AFTER our evaluation, since Corpus is modified
    result = model.evaluate(data_points=corpus.dev, gold_label_type=label_type, gold_label_dictionary=label_dict)
    print(result.detailed_results)

    # repeat this process on the test set
    test_predictions = predict(data_points=corpus.test, model=model, batch_size=32)
    corpus = load_corpus()
    test_scores = scorer_evaluate(corpus.test, test_predictions)
    test_scores.print_score_report()

    # check by evaluating with Flair
    test_result = model.evaluate(data_points=corpus.test, gold_label_type=label_type, gold_label_dictionary=label_dict)
    print(test_result.detailed_results)