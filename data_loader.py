from flair.data import Corpus
from flair.datasets import ColumnCorpus

def prepare_corpus(data_dir: str, train_file: str, dev_file: str, test_file: str, label_type: str) -> Corpus:
    """
    Prepares the corpus for training and evaluation.

    Args:
        data_dir (str): Directory containing the dataset files.
        train_file (str): Name of the training file.
        dev_file (str): Name of the development file.
        test_file (str): Name of the test file.
        label_type (str): Type of label (e.g., 'ner').

    Returns:
        Corpus: A Flair Corpus object containing the train, dev, and test datasets.
    """
    columns = {0: 'text', 1: label_type}

    corpus = ColumnCorpus(
        data_dir,
        columns,
        train_file=train_file,
        dev_file=dev_file,
        test_file=test_file,
    )

    return corpus