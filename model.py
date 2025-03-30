from typing import List, Dict, Any, Optional
from pathlib import Path

from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    TransformerWordEmbeddings
)
from flair.models import SequenceTagger
from flair.data import Dictionary

class NERModelBuilder:
    """Class to build and configure NER models using FLAIR."""
    
    @staticmethod
    def create_classic_flair_embeddings() -> StackedEmbeddings:
        """
        Create Flair + GloVe stacked embeddings.
        
        Returns:
            StackedEmbeddings: Stacked Flair and GloVe embeddings
        """
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
            FlairEmbeddings('news-forward'),
            FlairEmbeddings('news-backward'),
        ]
        return StackedEmbeddings(embeddings=embedding_types)
    
    @staticmethod
    def create_transformer_embeddings(model_name: str = 'distilbert-base-uncased',
                                      fine_tune: bool = False,
                                      use_context: bool = False) -> TransformerWordEmbeddings:
        """
        Create transformer embeddings.
        
        Args:
            model_name: Name of the transformer model
            fine_tune: Whether to fine-tune the transformer
            use_context: Whether to use document context
            
        Returns:
            TransformerWordEmbeddings: Configured transformer embeddings
        """
        return TransformerWordEmbeddings(
            model=model_name,
            layers="-1",
            subtoken_pooling="first",
            fine_tune=fine_tune,
            use_context=use_context,
        )
    
    @staticmethod
    def create_tagger(embeddings: TokenEmbeddings,
                      tag_dictionary: Dictionary,
                      tag_type: str = 'ner',
                      hidden_size: int = 256,
                      use_crf: bool = True,
                      use_rnn: bool = True,
                      rnn_layers: int = 1) -> SequenceTagger:
        """
        Create a sequence tagger model.
        
        Args:
            embeddings: Word embeddings to use
            tag_dictionary: Dictionary containing the tags
            tag_type: Type of tags
            hidden_size: Size of the hidden layers
            use_crf: Whether to use CRF
            use_rnn: Whether to use RNN
            rnn_layers: Number of RNN layers
            
        Returns:
            SequenceTagger: Configured sequence tagger model
        """
        # For transformer fine-tuning, we typically don't use CRF or RNN
        if isinstance(embeddings, TransformerWordEmbeddings) and embeddings.fine_tune:
            use_rnn = False
            reproject_embeddings = False
        else:
            reproject_embeddings = True
        
        return SequenceTagger(
            hidden_size=hidden_size,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=use_crf,
            use_rnn=use_rnn,
            rnn_layers=rnn_layers,
            reproject_embeddings=reproject_embeddings,
        )

class PartialEvaluationMetrics:
    """
    Class implementing various partial evaluation metrics for NER.
    Based on SemEval 2013 and other approaches for partial credit.
    """
    
    @staticmethod
    def exact_match(gold_entities, pred_entities):
        """Traditional exact match F1 score"""
        # Implementation will go here
        pass
    
    @staticmethod
    def left_boundary_match(gold_entities, pred_entities):
        """Left boundary match (good for relation extraction)"""
        # Implementation will go here
        pass
    
    @staticmethod
    def right_boundary_match(gold_entities, pred_entities):
        """Right boundary match (useful for missing/included preceding adjectives)"""
        # Implementation will go here
        pass
        
    @staticmethod
    def partial_match(gold_entities, pred_entities):
        """Partial match (any fragment overlap)"""
        # Implementation will go here
        pass
    
    @staticmethod
    def approximate_match(gold_entities, pred_entities, threshold=0.5):
        """Approximate match with configurable overlap threshold"""
        # Implementation will go here
        pass
    
    @staticmethod
    def fragment_percentage(gold_entities, pred_entities):
        """Calculate what percentage of each entity is correctly identified"""
        # Implementation will go here
        pass