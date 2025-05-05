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

import pickle

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

def generate_visualizations(scores, output_dir="charts", dataset_name=""):
    """Generates and saves bar charts of the evaluation metrics."""
    print(f"Generating visualizations for '{dataset_name}' in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid") 

    # --- Visualization 1: Metrics Comparison --
    metrics_data = {
        'Metric': ['Exact Match', 'Left Boundary', 'Right Boundary', 'Partial (Overlap)'],
        'Precision': [
            scores.precision(),
            scores.left_match_precision(),
            scores.right_match_precision(),
            scores.partial_match_precision()
        ],
        'Recall': [
            scores.recall(),
            scores.left_match_recall(),
            scores.right_match_recall(),
            scores.partial_match_recall()
        ],
        'F1 Score': [
            scores.f1_score(),
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
    ax_bar.set_title(f'Comparison of NER Evaluation Metrics ({dataset_name.capitalize()})', fontsize=14) 
    ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax_bar.legend(title='Score Type')
    ax_bar.grid(axis='y', linestyle='--')
    ax_bar.set_ylim(0, 105) # Set y-axis limit slightly above 100

    for container in ax_bar.containers:
        ax_bar.bar_label(container, fmt='%.1f', padding=3, fontsize=8)

    plt.tight_layout()
    filename_prefix = f"{dataset_name}_" if dataset_name else ""
    chart_path = os.path.join(output_dir, f"{filename_prefix}metrics_comparison_chart_seaborn.png")
    plt.savefig(chart_path)
    plt.close() # Close the figure to free memory
    print(f"Saved seaborn comparison chart to {chart_path}")

    # --- Visualization 2: Partial Match Credit Distribution ---
    print(f"Generating partial match credit distribution chart for '{dataset_name}'...")
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
    ax_credits.set_title(f'Distribution of Exact (1.0) vs. Partial (0.5) Credits ({dataset_name.capitalize()})', fontsize=14) 
    ax_credits.legend(title='Credit Type')
    ax_credits.grid(axis='y', linestyle='--')

    # Add value labels on top of bars
    for container in ax_credits.containers:
        ax_credits.bar_label(container, fmt='%d', padding=3, fontsize=9)

    plt.tight_layout()
    credit_chart_path = os.path.join(output_dir, f"{filename_prefix}partial_match_credit_distribution.png")
    plt.savefig(credit_chart_path)
    plt.close() 
    print(f"Saved partial match credit distribution chart to {credit_chart_path}")


def generate_comparison_visualization(dev_scores, test_scores, output_dir="comparison_charts"):
    """Generates and saves a bar chart comparing dev and test set metrics."""
    print(f"Generating comparison visualization in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    datasets = {'dev': dev_scores, 'test': test_scores}
    comparison_data = []

    for dataset_name, scores in datasets.items():
        metrics_data = {
            'Metric': ['Exact Match', 'Left Boundary', 'Right Boundary', 'Partial (Overlap)'],
            'Precision': [
                scores.precision(),
                scores.left_match_precision(),
                scores.right_match_precision(),
                scores.partial_match_precision()
            ],
            'Recall': [
                scores.recall(),
                scores.left_match_recall(),
                scores.right_match_recall(),
                scores.partial_match_recall()
            ],
            'F1 Score': [
                scores.f1_score(),
                scores.left_match_f1(),
                scores.right_match_f1(),
                scores.partial_match_f1()
            ]
        }
        df = pd.DataFrame(metrics_data)
        df_melted = df.melt(id_vars='Metric', var_name='Score Type', value_name='Score Value')
        df_melted['Dataset'] = dataset_name
        comparison_data.append(df_melted)

    df_comparison = pd.concat(comparison_data)
    df_comparison['Score Value'] = df_comparison['Score Value'].round(4) * 100

    g = sns.catplot(
        data=df_comparison, kind="bar",
        x="Metric", y="Score Value", hue="Dataset", col="Score Type",
        palette="viridis", height=6, aspect=.7 
    )

    g.set_axis_labels("Evaluation Metric Type", "Score (%)")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45, ha="right", fontsize=10)
    g.fig.suptitle('Comparison of Dev vs. Test NER Evaluation Metrics', y=1.03, fontsize=16) 
    g.despine(left=True) 
    g.set(ylim=(0, 105))

    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8)

    sns.move_legend(g, "upper left", bbox_to_anchor=(1.01, 1), title='Dataset')

    plt.tight_layout(rect=[0, 0, 0.95, 0.97]) 
    chart_path = os.path.join(output_dir, "dev_vs_test_metrics_comparison.png")
    plt.savefig(chart_path, bbox_inches='tight') 
    plt.close('all') 
    print(f"Saved comparison chart to {chart_path}")


if __name__ == "__main__":
    # load corpus
    corpus = load_corpus()

    # extract the labels from the corpus
    # TODO why is add_unk=True required?
    label_type = 'ner'
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=True)

    # load model from file
    model_path = "models/best-model.pt"  
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

    generate_visualizations(scores, output_dir="dev_charts", dataset_name="dev") 

    # dump scores to file
    dev_scores_dict = scores.get_score_dict() # Store dev scores dict
    with open("dev_scores.pkl", mode="wb") as file: # Use .pkl extension
        pickle.dump(dev_scores_dict, file)

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

    # write partial matches
    test_scores.write_partial_matches("predictions/partial_test_overlap.csv", match_type="overlap")
    test_scores.write_partial_matches("predictions/partial_test_left_bound.csv", match_type="left")
    test_scores.write_partial_matches("predictions/partial_test_right_bound.csv", match_type="right")

    generate_visualizations(test_scores, output_dir="test_charts", dataset_name="test")
    
    # dump scores to file
    test_scores_dict = test_scores.get_score_dict() # Store test scores dict
    with open("test_scores.pkl", mode="wb") as file: # Use .pkl extension
        pickle.dump(test_scores_dict, file)

    # Generate the comparison chart using the collected scores objects
    generate_comparison_visualization(scores, test_scores, output_dir="comparison_charts")