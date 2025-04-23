import os

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from tqdm import tqdm

from scorer import Scorer


def load_corpus() -> Corpus:
    # dataset format
    columns = {0: 'text', 1: 'ner'}

    # dataset location
    data_folder = 'broad_twitter_corpus'

    # load the data into a corpus
    corpus: Corpus = ColumnCorpus(data_folder=data_folder, column_format=columns, train_file='train.txt', test_file='test.txt', dev_file='dev.txt', column_delimiter="\t")

    return corpus


def train_model(corpus: Corpus) -> None:
    # TODO this model is from the flair tutorial and is not tuned
    embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            tag_format="BIO")

    trainer = ModelTrainer(tagger, corpus)

    trainer.train('resources/taggers/sota-ner-flair',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=150)


def grid_search(corpus: Corpus, params) -> None:
    pass


if __name__ == "__main__":
    # load corpus
    corpus = load_corpus()

    # extract the labels from the corpus
    label_type = 'ner'
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=True)

    # train model
    train_model(corpus)