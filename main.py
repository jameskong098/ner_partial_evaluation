import argparse
import os
from pathlib import Path

from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from data_loader import prepare_corpus
from model import NERModelBuilder, PartialEvaluationMetrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate NER models with partial evaluation metrics")
    
    # Data options
    parser.add_argument('--use_huggingface', action='store_true', help='Use HuggingFace datasets instead of local files')
    
    # Model options
    parser.add_argument('--model_type', choices=['classic', 'transformer'], default='classic',
                        help='Type of model to use: classic (Flair+GloVe) or transformer')
    parser.add_argument('--transformer_model', type=str, default='distilbert-base-uncased',
                        help='Transformer model to use if model_type is transformer')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune the transformer model')
    parser.add_argument('--use_context', action='store_true', help='Use document context for transformer')
    
    # Training options
    parser.add_argument('--output_dir', type=str, default='resources/taggers/twitter-ner',
                        help='Directory to save the model')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for training')
    parser.add_argument('--mini_batch_size', type=int, default=32,
                        help='Mini batch size for training')
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='Maximum number of training epochs')
    
    # Evaluation options
    parser.add_argument('--evaluate_only', action='store_true', 
                        help='Only run evaluation on existing model')
    parser.add_argument('--model_path', type=str, 
                        help='Path to existing model for evaluation')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading corpus...")
    corpus = prepare_corpus(args.use_huggingface)
    print(corpus)
    
    tag_type = 'ner'
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type, add_unk=False)
    print(tag_dictionary)
    
    if not args.evaluate_only:
        if args.model_type == 'classic':
            print("Creating classic Flair embeddings...")
            embeddings = NERModelBuilder.create_classic_flair_embeddings()
            learning_rate = args.learning_rate
        else:  
            print(f"Creating transformer embeddings with {args.transformer_model}...")
            embeddings = NERModelBuilder.create_transformer_embeddings(
                model_name=args.transformer_model,
                fine_tune=args.fine_tune,
                use_context=args.use_context
            )
            # Use a smaller learning rate for fine-tuning
            learning_rate = 5.0e-6 if args.fine_tune else 0.1
        
        # Create sequence tagger
        tagger = NERModelBuilder.create_tagger(
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=(not args.fine_tune)  # Don't use CRF when fine-tuning transformers
        )
        
        trainer = ModelTrainer(tagger, corpus)
        
        print(f"Starting training with learning_rate={learning_rate}...")
        trainer.train(
            base_path=args.output_dir,
            learning_rate=learning_rate,
            mini_batch_size=args.mini_batch_size,
            max_epochs=args.max_epochs,
        )

    print("Evaluating model...")
    model_path = args.model_path if args.evaluate_only else f"{args.output_dir}/best-model.pt"

    tagger = SequenceTagger.load(model_path)
    
    # TODO: Implement custom evaluation with partial metrics
    # For now, use the standard evaluation
    result = tagger.evaluate(corpus.test, gold_label_type='ner')
    print(result.detailed_results)
    
    # TODO: Add partial evaluation metrics implementation
    print("Note: Partial evaluation metrics will be implemented in the future")

if __name__ == "__main__":
    main()