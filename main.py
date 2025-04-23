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
corpus: Corpus = ColumnCorpus(data_folder=data_folder, column_format=columns, train_file='btc.train', test_file='btc.test', dev_file='use_this.dev', column_delimiter="\t")

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

# TODO before next train, try adding tag_format = 'BIO' - by default, this is using BIOES
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


def perform_error_analysis(model, corpus, output_file='error_analysis.txt'):
    """Perform qualitative error analysis on partial matches and write to file."""
    with open(output_file, 'w') as f:
        f.write("=== NER Partial Match Error Analysis ===\n\n")
        
        for i, sentence in tqdm(enumerate(corpus.dev), total=len(corpus.dev), desc="Analyzing errors"):
            # Get gold standard mentions
            gold_mentions = Scorer.create_mentions(sentence.get_labels())
            
            # Get model predictions
            prediction_sentence = Sentence(sentence.text) # I think this causes different tokenization in the predictions
            model.predict(prediction_sentence)
            predicted_mentions = Scorer.create_mentions(prediction_sentence.get_labels())
            
            # Skip if perfect match or no mentions
            if (set(gold_mentions) == set(predicted_mentions)) or not (gold_mentions and predicted_mentions):
                continue
                
            # Create individual scorers for this sentence
            sentence_scorer = Scorer(gold_mentions, predicted_mentions)
            
            # Skip if no partial matches at all (using partial_match_f1 as an indicator)
            if sentence_scorer.partial_match_f1() <= 0 or sentence_scorer.f1_score() == 1.0:
                continue
            
            f.write(f"Sentence {i}: {sentence.text}\n")
            f.write("-" * 80 + "\n")
            
            # Compare gold and predictions
            f.write("GOLD MENTIONS:\n")
            f.write("Mention object: " + str(gold_mentions) + "\n")
            for mention in gold_mentions:
                try:
                    # Ensure start and end are integers
                    start = int(mention.start)
                    end = int(mention.end)
                    text = mention.text
                    # Use entity_type instead of tag (or get any available attribute safely)
                    entity_type = getattr(mention, 'entity_type', getattr(mention, 'type', getattr(mention, 'label', '???')))
                    f.write(f"  - \"{text}\" [{entity_type}] ({start}:{end})\n")
                except (ValueError, TypeError, IndexError, AttributeError) as e:
                    f.write(f"  - Error extracting text: {str(e)} - Raw mention: {mention}\n")
            
            f.write("\nPREDICTIONS:\n")
            for mention in predicted_mentions:
                try:
                    # Ensure start and end are integers
                    start = int(mention.start)
                    end = int(mention.end)
                    text = mention.text
                    # Use entity_type instead of tag (or get any available attribute safely)
                    entity_type = getattr(mention, 'entity_type', getattr(mention, 'type', getattr(mention, 'label', '???')))
                    f.write(f"  - \"{text}\" [{entity_type}] ({start}:{end})\n")
                except (ValueError, TypeError, IndexError, AttributeError) as e:
                    f.write(f"  - Error extracting text: {str(e)} - Raw mention: {mention}\n")
            
            f.write("\nPARTIAL MATCH METRICS:\n")
            f.write(f"Exact Match F1: {sentence_scorer.f1_score():.4f}\n")
            f.write(f"Left Boundary Match F1: {sentence_scorer.left_match_f1():.4f}\n")
            f.write(f"Right Boundary Match F1: {sentence_scorer.right_match_f1():.4f}\n")
            f.write(f"Partial Overlap F1: {sentence_scorer.partial_match_f1():.4f}\n")
            f.write(f"Overlap Percentage F1: {sentence_scorer.overlap_f1():.4f}\n")
            
            f.write("\nDETAILED ANALYSIS:\n")
            for gold in gold_mentions:
                try:
                    gold_start = int(gold.start)
                    gold_end = int(gold.end)
                    gold_text = gold.text
                    gold_type = getattr(gold, 'entity_type', getattr(gold, 'type', getattr(gold, 'label', '???')))
                    
                    for pred in predicted_mentions:
                        try:
                            pred_start = int(pred.start)
                            pred_end = int(pred.end)
                            pred_text = pred.text
                            pred_type = getattr(pred, 'entity_type', getattr(pred, 'type', getattr(pred, 'label', '???')))
                            
                            # Check if there is any overlap
                            if max(gold_start, pred_start) < min(gold_end, pred_end):
                                overlap_start = max(gold_start, pred_start)
                                overlap_end = min(gold_end, pred_end)
                                overlap_text = sentence.text[overlap_start:overlap_end]
                                overlap_chars = overlap_end - overlap_start
                                gold_length = gold_end - gold_start
                                pred_length = pred_end - pred_start
                                overlap_gold_percent = (overlap_chars / gold_length) * 100
                                overlap_pred_percent = (overlap_chars / pred_length) * 100
                                
                                f.write(f"\n  Gold: \"{gold_text}\" [{gold_type}] vs Pred: \"{pred_text}\" [{pred_type}]\n")
                                f.write(f"  Overlap: \"{overlap_text}\" ({overlap_chars} chars)\n")
                                f.write(f"  Overlap covers {overlap_gold_percent:.1f}% of gold and {overlap_pred_percent:.1f}% of prediction\n")
                                
                                # Report which boundaries match
                                if gold_start == pred_start:
                                    f.write("  LEFT BOUNDARY MATCH: Yes\n")
                                else:
                                    f.write(f"  LEFT BOUNDARY MISMATCH: Gold={gold_start}, Pred={pred_start}, Offset={pred_start-gold_start}\n")
                                
                                if gold_end == pred_end:
                                    f.write("  RIGHT BOUNDARY MATCH: Yes\n")
                                else:
                                    f.write(f"  RIGHT BOUNDARY MISMATCH: Gold={gold_end}, Pred={pred_end}, Offset={pred_end-gold_end}\n")
                        except (ValueError, TypeError, IndexError) as e:
                            f.write(f"  Error processing prediction: {str(e)}\n")
                except (ValueError, TypeError, IndexError) as e:
                    f.write(f"  Error processing gold mention: {str(e)}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("\nAnalysis complete. Examined all partial matches in the dev set.")
    
    print(f"Error analysis written to {output_file}")


# result = model.evaluate(data_points=corpus.dev, gold_label_type='ner', gold_label_dictionary=label_dict)

# print("Flair built-in F1:", result.main_score)

# scores = Scorer([], [])
# for i, sentence in tqdm(enumerate(corpus.dev), total=len(corpus.dev)):
#     reference = Scorer.create_mentions(sentence.get_labels())
#     unlabeled = Sentence(sentence.text)
#     model.predict(unlabeled)
#     predictions = Scorer.create_mentions(unlabeled.get_labels())
#     scores.merge(Scorer(reference, predictions))

# print("\n===== NER Evaluation Results =====")
# print("Exact Match Metrics (Traditional):")
# print(f"Custom Scorer F1: {scores.f1_score():.4f}")
# print(f"Precision: {scores.precision():.4f}")
# print(f"Recall: {scores.recall():.4f}")

# print("\nPartial Match Metrics:")
# print(f"Left Boundary Match F1: {scores.left_match_f1():.4f}")
# print(f"Right Boundary Match F1: {scores.right_match_f1():.4f}")
# print(f"Partial Overlap F1: {scores.partial_match_f1():.4f}")
# print(f"Overlap Percentage F1: {scores.overlap_f1():.4f}")

# print("\nDetailed Partial Match Metrics:")
# print(f"Left Match - Precision: {scores.left_match_precision():.4f}, Recall: {scores.left_match_recall():.4f}")
# print(f"Right Match - Precision: {scores.right_match_precision():.4f}, Recall: {scores.right_match_recall():.4f}")
# print(f"Partial Match - Precision: {scores.partial_match_precision():.4f}, Recall: {scores.partial_match_recall():.4f}")
# print(f"Overlap - Precision: {scores.overlap_precision():.4f}, Recall: {scores.overlap_recall():.4f}")

# perform_error_analysis(model, corpus, output_file='ner_error_analysis.txt')
perform_error_analysis(model, corpus, output_file='ner_error_analysis_2.txt')