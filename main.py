import os

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from tqdm import tqdm

from scorer import Scorer

# format of dataset
columns = {0: 'text', 1: 'ner'}

# location of data files
data_folder = 'broad_twitter_corpus'

# load the data into a Corpus of Sentences
# removed 1 specific tweet from the dev set which was causing an error while loading the data
corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='btc.train', test_file='btc.test', dev_file='use_this.dev', column_delimiter="\t")

# as a test to start out, using smaller train set to speed up training
# corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='a.conll', test_file='f.conll', dev_file='hfixed.conll')

# extract the labels from the corpus
label_type = 'ner'
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=True)
print(label_dict)

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
                        tag_type=label_type)

trainer = ModelTrainer(tagger, corpus)

# trainer.train('resources/taggers/sota-ner-flair',
#               learning_rate=0.1,
#               mini_batch_size=32,
#               max_epochs=150)

model = SequenceTagger.load(os.path.join('resources', 'taggers', 'sota-ner-flair', 'final-model.pt'))

model.evaluate(data_points=corpus.dev, gold_label_type='ner', gold_label_dictionary=label_dict)

scores = Scorer([], [])
for i, sentence in tqdm(enumerate(corpus.dev), total=len(corpus.dev)):
    reference = Scorer.create_mentions(sentence.get_labels())
    unlabeled = Sentence(sentence.text)
    model.predict(unlabeled)
    predictions = Scorer.create_mentions(unlabeled.get_labels())
    scores.merge(Scorer(reference, predictions))

print("F1:", scores.f1_score())