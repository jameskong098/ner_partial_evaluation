from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from datasets import load_dataset

class BroadTwitterCorpusLoader:
    """
    Class to load the Broad Twitter Corpus for NER training and evaluation.
    Handles both loading from local files or from HuggingFace datasets.
    """
    
    @staticmethod
    def load_local_corpus(data_folder: str = "dataset") -> Corpus:
        """
        Load corpus from local CoNLL files in subdirectories.
        Selects a single file from each directory to avoid multiple file loading issues.
        
        Args:
            data_folder: Path to the folder containing the dataset files
            
        Returns:
            A Flair Corpus object
        """
        columns = {0: "text", 1: "ner"}
        
        train_file = "train/b.conll" 
        dev_file = "dev/h.conll"
        test_file = "test/f.conll"
        
        return ColumnCorpus(
            data_folder=data_folder,
            column_format=columns,
            train_file=train_file,
            test_file=test_file,
            dev_file=dev_file,
        )
    
    @staticmethod
    def load_from_huggingface() -> Corpus:
        """
        Load corpus from HuggingFace datasets library.
        
        Returns:
            A Flair Corpus object
        """
        # Load from HuggingFace datasets
        from datasets import load_dataset
        
        dataset = load_dataset("broad_twitter_corpus.py")
        
        # Convert HuggingFace dataset to Flair Corpus
        return BroadTwitterCorpusLoader._convert_to_flair_corpus(dataset)
    
    @staticmethod
    def _convert_to_flair_corpus(dataset) -> Corpus:
        """
        Convert HuggingFace dataset to Flair Corpus format.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            A Flair Corpus object
        """
        from flair.data import Corpus, Sentence, Token, Span
        from flair.datasets import FlairDataset
        
        class HuggingFaceDataset(FlairDataset):
            def __init__(self, hf_dataset):
                self.hf_dataset = hf_dataset
                
            def __getitem__(self, index):
                example = self.hf_dataset[index]
                sentence = Sentence()
                
                for token_text, ner_tag in zip(example['tokens'], example['ner_tags']):
                    token = Token(token_text)
                    token.add_tag('ner', ner_tag)
                    sentence.add_token(token)
                    
                return sentence
            
            def __len__(self):
                return len(self.hf_dataset)
        
        # Create Flair datasets for each split
        train_dataset = HuggingFaceDataset(dataset['train'])
        dev_dataset = HuggingFaceDataset(dataset['validation'])
        test_dataset = HuggingFaceDataset(dataset['test'])
        
        return Corpus(train=train_dataset, dev=dev_dataset, test=test_dataset)

def prepare_corpus(use_huggingface: bool = False) -> Corpus:
    """
    Main function to load and prepare the corpus.
    
    Args:
        use_huggingface: Whether to load from HuggingFace or local files
        
    Returns:
        A Flair Corpus object
    """
    if use_huggingface:
        return BroadTwitterCorpusLoader.load_from_huggingface()
    else:
        return BroadTwitterCorpusLoader.load_local_corpus()