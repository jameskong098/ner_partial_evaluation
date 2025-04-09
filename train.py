from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

if __name__ == "__main__":
    # define columns
    columns = {0: "text", 1: "ner"}

    # this is the folder in which train, test and dev files reside
    data_folder = "tweetner7"

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file="train.txt",
                                  test_file="test.txt",
                                  dev_file="dev.txt", column_delimiter="\t")

    label_type = "ner"

    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)

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