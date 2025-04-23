from flair.data import Label
from typing_extensions import NamedTuple
from typing import Sequence, Dict, Tuple, List, Set, Optional

# for testing purposes
from flair.data import Sentence
from flair.nn import Classifier

class Mention(NamedTuple):
    """
    Start index inclusive, end index exclusive.
    """
    entity_type: str
    start: int
    end: int
    text: str

class Scorer:
    def __init__(self, reference: Sequence[Mention], predictions: Sequence[Mention]) -> None:
        """
        Compute counts necessary for easily calculating metrics.
        """
        reference_set = set(reference)
        predictions_set = set(predictions)
        self.true_positives = len(reference_set & predictions_set)
        self.false_positives = len(predictions_set - reference_set)
        self.false_negatives = len(reference_set - predictions_set)
        
        # Store raw mentions for partial matching
        self.reference = list(reference)
        self.predictions = list(predictions)
        
        self.left_match_tp = 0
        self.right_match_tp = 0
        self.partial_match_tp = 0
        self.type_match_tp = 0
        self.overlap_scores = 0.0
        
        self._compute_partial_matches()
        
    def _compute_partial_matches(self) -> None:
        """
        Compute all the partial match metrics between reference and predictions.
        """
        # Track which references have been matched to avoid double counting
        matched_refs: Dict[int, Set[str]] = {i: set() for i in range(len(self.reference))}
        
        for pred in self.predictions:
            best_overlap = 0
            best_ref_idx = -1
            
            # Find the best matching reference for this prediction
            for i, ref in enumerate(self.reference):
                # Skip if entity types don't match and we're not doing type-relaxed matching
                if pred.entity_type != ref.entity_type:
                    continue
                    
                # Check left boundary match
                if int(pred.start) == int(ref.start):
                    if 'left' not in matched_refs[i]:
                        self.left_match_tp += 1
                        matched_refs[i].add('left')
                
                # Check right boundary match
                if int(pred.end) == int(ref.end):
                    if 'right' not in matched_refs[i]:
                        self.right_match_tp += 1
                        matched_refs[i].add('right')
                
                # Check for any overlap between the spans
                if max(int(pred.start), int(ref.start)) < min(int(pred.end), int(ref.end)):
                    if 'partial' not in matched_refs[i]:
                        self.partial_match_tp += 1
                        matched_refs[i].add('partial')
                    
                    # Calculate overlap percentage
                    overlap_start = max(int(pred.start), int(ref.start))
                    overlap_end = min(int(pred.end), int(ref.end))
                    overlap_length = overlap_end - overlap_start
                    
                    union_length = max(int(pred.end), int(ref.end)) - min(int(pred.start), int(ref.start))
                    overlap_ratio = overlap_length / union_length
                    
                    # Track the best overlap for this prediction
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_ref_idx = i
            
            # Add the best overlap score if there was a match
            if best_ref_idx >= 0 and 'overlap' not in matched_refs[best_ref_idx]:
                self.overlap_scores += best_overlap
                matched_refs[best_ref_idx].add('overlap')
        
        # Count type matches (same entity type, regardless of spans)
        pred_types = {p.entity_type for p in self.predictions}
        ref_types = {r.entity_type for r in self.reference}
        
        for pred_type in pred_types:
            if pred_type in ref_types:
                self.type_match_tp += 1

    def precision(self) -> float:
        """
        Finds exact mention-level precision using pre-computed counts.
        """
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self) -> float:
        """
        Finds exact mention-level recall using pre-computed counts.
        """
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def f1_score(self) -> float:
        """
        Finds exact mention-level F1 using pre-computed counts.
        """
        if self.precision() + self.recall() == 0:
            return 0.0
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall())
    
    def left_match_precision(self) -> float:
        """
        Precision for left boundary matches.
        """
        if len(self.predictions) == 0:
            return 0.0
        return self.left_match_tp / len(self.predictions)
    
    def left_match_recall(self) -> float:
        """
        Recall for left boundary matches.
        """
        if len(self.reference) == 0:
            return 0.0
        return self.left_match_tp / len(self.reference)
    
    def left_match_f1(self) -> float:
        """
        F1 score for left boundary matches.
        """
        precision = self.left_match_precision()
        recall = self.left_match_recall()
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)
    
    def right_match_precision(self) -> float:
        """
        Precision for right boundary matches.
        """
        if len(self.predictions) == 0:
            return 0.0
        return self.right_match_tp / len(self.predictions)
    
    def right_match_recall(self) -> float:
        """
        Recall for right boundary matches.
        """
        if len(self.reference) == 0:
            return 0.0
        return self.right_match_tp / len(self.reference)
    
    def right_match_f1(self) -> float:
        """
        F1 score for right boundary matches.
        """
        precision = self.right_match_precision()
        recall = self.right_match_recall()
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)
    
    def partial_match_precision(self) -> float:
        """
        Precision for partial span overlap.
        """
        if len(self.predictions) == 0:
            return 0.0
        return self.partial_match_tp / len(self.predictions)
    
    def partial_match_recall(self) -> float:
        """
        Recall for partial span overlap.
        """
        if len(self.reference) == 0:
            return 0.0
        return self.partial_match_tp / len(self.reference)
    
    def partial_match_f1(self) -> float:
        """
        F1 score for partial span overlap.
        """
        precision = self.partial_match_precision()
        recall = self.partial_match_recall()
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)
    
    def overlap_precision(self) -> float:
        """
        Weighted precision based on overlap percentage.
        """
        if len(self.predictions) == 0:
            return 0.0
        return self.overlap_scores / len(self.predictions)
    
    def overlap_recall(self) -> float:
        """
        Weighted recall based on overlap percentage.
        """
        if len(self.reference) == 0:
            return 0.0
        return self.overlap_scores / len(self.reference)
    
    def overlap_f1(self) -> float:
        """
        F1 score weighted by overlap percentage.
        """
        precision = self.overlap_precision()
        recall = self.overlap_recall()
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def merge(self, other_scorer: "Scorer") -> None:
        """
        Adds the other Scorer's counts to this one.
        """
        self.true_positives += other_scorer.true_positives
        self.false_positives += other_scorer.false_positives
        self.false_negatives += other_scorer.false_negatives
        
        self.left_match_tp += other_scorer.left_match_tp
        self.right_match_tp += other_scorer.right_match_tp
        self.partial_match_tp += other_scorer.partial_match_tp
        self.type_match_tp += other_scorer.type_match_tp
        self.overlap_scores += other_scorer.overlap_scores
        
        # Append the raw mentions for potential recomputation
        self.reference.extend(other_scorer.reference)
        self.predictions.extend(other_scorer.predictions)


    @staticmethod
    def create_mentions(labels: Sequence[Label]) -> list[Mention]:
        """
        Given a list of flair Labels for a sentence, return a list of mentions for each labeled span.
        Assumes that the spans provided are valid.
        """
        mentions = []
        for label in labels:
            entity_type = label.value
            start = label.unlabeled_identifier.split(":")[0].split("[")[1]
            end = label.unlabeled_identifier.split(":")[1].split("]")[0]
            text = label.data_point.text
            mentions.append(Mention(entity_type, start, end, text))
        return mentions
    

if __name__ == "__main__":
    # simple test case
    sentence = Sentence('George Washington went to Washington.')

    tagger = Classifier.load('ner-fast')

    tagger.predict(sentence)

    print(sentence)

    gold = [Mention(entity_type='PER', start='0', end='2', text='George Washington'), Mention(entity_type='LOC', start='4', end='5', text='Washington')]

    print(Scorer.create_mentions(sentence.get_labels()))

    test = Scorer(gold, Scorer.create_mentions(sentence.get_labels()))

    print(test.f1_score())