import os
from pathlib import Path

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, CharacterEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from tqdm import tqdm

from scorer import Scorer


def load_corpus(data_folder: str = 'broad_twitter_corpus', train_file='btc.train', test_file='btc.test', dev_file='use_this.dev', delim = "\t", debug: bool = False) -> Corpus:
    # dataset format
    columns = {0: 'text', 1: 'ner'}

    # load the data into a corpus
    corpus: Corpus = ColumnCorpus(data_folder=data_folder, column_format=columns, train_file=train_file, test_file=test_file, dev_file=dev_file, column_delimiter=delim)

    if debug:
        return corpus.downsample(0.1)
    
    return corpus


def train_model(corpus: Corpus) -> None:
    # TODO this model is from the flair tutorial and is not tuned

    embedding_types = [
        WordEmbeddings('twitter'),
        CharacterEmbeddings(),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            tag_format="BIO",
                            use_crf=True,
                            use_rnn=True,)

    trainer = ModelTrainer(tagger, corpus)

    # Define the output path as a Path object
    output_path = Path('resources/taggers/sota-ner-flair')
    output_path.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

    trainer.train(output_path, # Pass the Path object here
                  monitor_test=True,
                  patience=3,
                  anneal_with_restarts=True,
                  learning_rate=0.05,
                  mini_batch_size=16,
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