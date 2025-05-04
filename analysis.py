import os

from train import load_corpus
from flair.models import SequenceTagger
from flair.datasets import DataLoader, FlairDatapointDataset
from scorer import Scorer
from flair.data import Sentence
from tqdm import tqdm

from collections import Counter

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns 

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

def generate_visualizations(scores, output_dir="charts"):
    """Generates and saves bar charts of the evaluation metrics."""
    print(f"Generating visualizations in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid") 

    # --- Visualization 1: Metrics Comparison --
    metrics_data = {
        'Metric': ['Exact Match', 'Exact Boundary', 'Left Boundary', 'Right Boundary', 'Partial (Overlap)'],
        'Precision': [
            scores.precision(),
            scores.exact_boundary_precision(),
            scores.left_match_precision(),
            scores.right_match_precision(),
            scores.partial_match_precision()
        ],
        'Recall': [
            scores.recall(),
            scores.exact_boundary_recall(),
            scores.left_match_recall(),
            scores.right_match_recall(),
            scores.partial_match_recall()
        ],
        'F1 Score': [
            scores.f1_score(),
            scores.exact_boundary_f1_score(),
            scores.left_match_f1(),
            scores.right_match_f1(),
            scores.partial_match_f1()
        ]
    }
    df = pd.DataFrame(metrics_data)
    df_melted = df.melt(id_vars='Metric', var_name='Score Type', value_name='Score Value')
    df_melted['Score Value'] = df_melted['Score Value'].round(4) * 100 

    # --- Bar Chart (Seaborn) ---
    plt.figure(figsize=(12, 7)) 
    ax_bar = sns.barplot(x='Metric', y='Score Value', hue='Score Type', data=df_melted, palette='viridis')

    ax_bar.set_xlabel('Evaluation Metric Type', fontsize=12)
    ax_bar.set_ylabel('Score (%)', fontsize=12)
    ax_bar.set_title('Comparison of NER Evaluation Metrics', fontsize=14)
    ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax_bar.legend(title='Score Type')
    ax_bar.grid(axis='y', linestyle='--')
    ax_bar.set_ylim(0, 105) # Set y-axis limit slightly above 100

    for container in ax_bar.containers:
        ax_bar.bar_label(container, fmt='%.1f', padding=3, fontsize=8)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "metrics_comparison_chart_seaborn.png") 
    plt.savefig(chart_path)
    plt.close() # Close the figure to free memory
    print(f"Saved seaborn comparison chart to {chart_path}")

    # --- Visualization 2: Partial Match Credit Distribution ---
    print("Generating partial match credit distribution chart...")
    credit_data = []

    # Process Overlap Credits
    overlap_exact = sum(1 for _, _, credit in scores.overlap_credit_list if credit == 1.0)
    overlap_partial = sum(1 for _, _, credit in scores.overlap_credit_list if credit == 0.5)
    credit_data.append({'Match Type': 'Overlap', 'Credit Type': 'Exact (1.0)', 'Count': overlap_exact})
    credit_data.append({'Match Type': 'Overlap', 'Credit Type': 'Partial (0.5)', 'Count': overlap_partial})

    # Process Left Boundary Credits
    left_exact = sum(1 for _, _, credit in scores.left_credit_list if credit == 1.0)
    left_partial = sum(1 for _, _, credit in scores.left_credit_list if credit == 0.5)
    credit_data.append({'Match Type': 'Left Boundary', 'Credit Type': 'Exact (1.0)', 'Count': left_exact})
    credit_data.append({'Match Type': 'Left Boundary', 'Credit Type': 'Partial (0.5)', 'Count': left_partial})

    # Process Right Boundary Credits
    right_exact = sum(1 for _, _, credit in scores.right_credit_list if credit == 1.0)
    right_partial = sum(1 for _, _, credit in scores.right_credit_list if credit == 0.5)
    credit_data.append({'Match Type': 'Right Boundary', 'Credit Type': 'Exact (1.0)', 'Count': right_exact})
    credit_data.append({'Match Type': 'Right Boundary', 'Credit Type': 'Partial (0.5)', 'Count': right_partial})

    df_credits = pd.DataFrame(credit_data)

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    ax_credits = sns.barplot(x='Match Type', y='Count', hue='Credit Type', data=df_credits, palette='muted')

    ax_credits.set_xlabel('Partial Match Strategy', fontsize=12)
    ax_credits.set_ylabel('Number of Matches', fontsize=12)
    ax_credits.set_title('Distribution of Exact (1.0) vs. Partial (0.5) Credits', fontsize=14)
    ax_credits.legend(title='Credit Type')
    ax_credits.grid(axis='y', linestyle='--')

    # Add value labels on top of bars
    for container in ax_credits.containers:
        ax_credits.bar_label(container, fmt='%d', padding=3, fontsize=9)

    plt.tight_layout()
    credit_chart_path = os.path.join(output_dir, "partial_match_credit_distribution.png")
    plt.savefig(credit_chart_path)
    plt.close() 
    print(f"Saved partial match credit distribution chart to {credit_chart_path}")

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
    os.makedirs("predictions", exist_ok=True) 
    scores.write_partial_matches("predictions/partial_dev_overlap.csv", match_type="overlap")
    scores.write_partial_matches("predictions/partial_dev_left_bound.csv", match_type="left")
    scores.write_partial_matches("predictions/partial_dev_right_bound.csv", match_type="right")

    generate_visualizations(scores)

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