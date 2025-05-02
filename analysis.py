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


# def write_conll(sentences, path):
#     """
#     Writes predictions to the specified path in CONLL format
#     """
#     for sentence in sentences:
#         for token in sentence:
#             if token.to_dict("ner")["labels"] == []:
#                 token.add_label(typename = "ner", value = "O") 

#     with open(path, mode="w", encoding="utf8") as file:
#         for sentence in sentences:
#             for i, token in enumerate(sentence):
#                 file.write(token.text)
#                 file.write("\t")
#                 file.write(token.to_dict("ner")["labels"][0]["value"])
#                 file.write("\n")
#             file.write("\n")


# # the accuracy found here is different than what Flair reports
# def check_accuracy(gold, pred):
#     """
#     Finds the accuracy of CONLL format tagged data in pred, comparing to gold
#     """
#     correct = 0
#     total = 0
#     with open(gold, encoding="utf8") as reference, open(pred, encoding="utf8") as prediction:
#         for line1, line2 in zip(reference, prediction):
#             if not (line1.strip("\n") == "" or line2.strip("\n") == ""):
#                 gold_tok = line1.rstrip().split("\t")[1]
#                 pred_tok = line2.rstrip().split("\t")[1]
#                 if gold_tok == pred_tok:
#                     correct +=1
#                 total += 1

#     # compute microavg acc
#     print(f"Accuracy: {correct/total}")


if __name__ == "__main__":
    # load corpus
    corpus = load_corpus()

    # extract the labels from the corpus
    # TODO why is add_unk=True required?
    label_type = 'ner'
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=True)

    # load model from file
    model = SequenceTagger.load(os.path.join('model', 'best-model.pt'))

    # evaluate with flair
    result = model.evaluate(data_points=corpus.dev, gold_label_type=label_type, gold_label_dictionary=label_dict)
    print(result.detailed_results)

    # evaluate with our Scorer
    predictions = predict(data_points=corpus.dev, model=model, batch_size=32)   

    # re-tagging the sentences means we lose the original dev tags
    # for now just reload the corpus, relatively fast
    corpus = load_corpus()
    scores = scorer_evaluate(corpus.dev, predictions)
    scores.print_score_report()
