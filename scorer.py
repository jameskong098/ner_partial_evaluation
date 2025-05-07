from flair.data import Label
from typing_extensions import NamedTuple
from typing import Sequence, Dict, Tuple, List, Set, Optional


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

        # strict evaluation
        self.true_positives = len(reference_set & predictions_set)
        self.false_positives = len(predictions_set - reference_set)
        self.false_negatives = len(reference_set - predictions_set)
        
        # Store raw mentions for partial matching
        self.reference = list(reference)
        self.predictions = list(predictions)

        self.possible = len(self.reference) # used for recall
        self.actual = len(self.predictions) # used for precision
        
        # Calculate Left Boundary Matches (Boundary + Type)
        left_reference_set = set([(mention.start, mention.entity_type) for mention in reference])
        left_predictions_set = set([(mention.start, mention.entity_type) for mention in predictions])
        self.left_match_tp = len(left_reference_set & left_predictions_set)
        self.left_match_fp = len(left_predictions_set - left_reference_set)
        self.left_match_fn = len(left_reference_set - left_predictions_set)

        # Calculate Right Boundary Matches (Boundary + Type)
        right_reference_set = set([(mention.end, mention.entity_type) for mention in reference])
        right_predictions_set = set([(mention.end, mention.entity_type) for mention in predictions])
        self.right_match_tp = len(right_reference_set & right_predictions_set)
        self.right_match_fp = len(right_predictions_set - right_reference_set)
        self.right_match_fn = len(right_reference_set - right_predictions_set)

        self.partial_match_tp, self.partial_match_fp, self.partial_match_fn, self.overlap_credit_list = self.__count_partial_matches()

        # build left/right credit lists for CSV output (credit based on exact vs partial boundary match)
        self.left_credit_list = []
        for ref in self.reference:
            for pred in self.predictions:
                # Check for left boundary and type match
                if ref.start == pred.start and ref.entity_type == pred.entity_type:
                    # Assign credit: 1.0 if ends also match (exact), 0.5 otherwise
                    if ref.end == pred.end:
                        credit = 1.0
                    else:
                        credit = 0.5
                        self.left_match_tp -= 0.5 # partial matches worth half credit towards score
                    self.left_credit_list.append((ref, pred, credit))

        self.right_credit_list = []
        for ref in self.reference:
            for pred in self.predictions:
                # Check for right boundary and type match
                if ref.end == pred.end and ref.entity_type == pred.entity_type:
                    # Assign credit: 1.0 if starts also match (exact), 0.5 otherwise
                    if ref.start == pred.start:
                        credit = 1.0
                    else:
                        credit = 0.5
                        self.right_match_tp -= 0.5 # partial matches worth half credit towards score
                    self.right_credit_list.append((ref, pred, credit))

    def __count_partial_matches(self) -> tuple[int, int, int]:
        partial_match_tp = 0
        partial_match_fp = 0
        partial_match_fn = 0
        partial_credit_list = []

        unmatched_references = list(self.reference)
        matched_predictions = set()
        predictions_to_match = list(self.predictions)
        
        # Prioritize exact matches first to prevent them from being counted as partial (0.5)
        remaining_predictions = []
        temp_unmatched_references = []
        
        # First pass: Exact matches (credit 1.0)
        for prediction in predictions_to_match:
            found_exact_match = False
            temp_refs_after_pass = []
            for reference in unmatched_references:
                # Check for exact match (start, end, type)
                if prediction.start == reference.start and prediction.end == reference.end and prediction.entity_type == reference.entity_type and not found_exact_match:
                    partial_match_tp += 1.0
                    partial_credit_list.append((reference, prediction, 1.0))
                    matched_predictions.add(prediction)
                    found_exact_match = True
                    # Don't add this reference to the next pass list
                else:
                    temp_refs_after_pass.append(reference) # Keep reference for next pass if not exactly matched
            
            if not found_exact_match:
                remaining_predictions.append(prediction) # Keep prediction for partial match pass
            unmatched_references = temp_refs_after_pass # Update references for the next prediction in this pass

        # Second pass: Overlapping matches (credit 0.5) - Use remaining predictions and references
        temp_unmatched_references = list(unmatched_references) # Reset list for the second pass
        final_unmatched_references = []

        for prediction in remaining_predictions:
            found_overlap_match = False
            temp_refs_after_pass = []
            for reference in temp_unmatched_references:
                 # Check for overlap (any overlap, same type) and ensure not already matched
                if self.__has_overlap(prediction, reference) and prediction.entity_type == reference.entity_type and not found_overlap_match:
                    partial_match_tp += 0.5
                    partial_credit_list.append((reference, prediction, 0.5))
                    matched_predictions.add(prediction)
                    found_overlap_match = True
                     # Don't add this reference to the final unmatched list
                else:
                    temp_refs_after_pass.append(reference) # Keep reference if not matched in this step

            if not found_overlap_match:
                 # This prediction is a false positive for overlap matching
                 pass # FP is calculated later based on `matched_predictions` set
            
            # Update reference list for the next prediction in *this* pass
            # A reference can only be matched once (either exactly or partially)
            temp_unmatched_references = temp_refs_after_pass 

        # References remaining after both passes are False Negatives for overlap matching
        final_unmatched_references = temp_unmatched_references

        # Calculate FP and FN based on the matching process
        partial_match_fp = len(set(self.predictions) - matched_predictions)
        partial_match_fn = len(final_unmatched_references)

        # Ensure TP doesn't exceed the number of predictions or references
        partial_match_tp = min(partial_match_tp, self.actual)
        partial_match_tp = min(partial_match_tp, self.possible)


        return partial_match_tp, partial_match_fp, partial_match_fn, partial_credit_list

    def __has_overlap(self, first: Mention, second: Mention) -> bool:
        """
        Returns True if the two given mentions have overlapping spans.
        """
        return max(int(first.start), int(second.start)) < min(int(first.end), int(second.end))

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
        if self.left_match_tp + self.left_match_fp == 0:
            return 0.0
        return self.left_match_tp / self.actual
    
    def left_match_recall(self) -> float:
        """
        Recall for left boundary matches.
        """
        if self.left_match_tp + self.left_match_fn == 0:
            return 0.0
        return self.left_match_tp / self.possible
    
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
        Precision for left boundary matches.
        """
        if self.right_match_tp + self.right_match_fp == 0:
            return 0.0
        return self.right_match_tp / self.actual
    
    def right_match_recall(self) -> float:
        """
        Recall for left boundary matches.
        """
        if self.right_match_tp + self.right_match_fn == 0:
            return 0.0
        return self.right_match_tp / self.possible
    
    def right_match_f1(self) -> float:
        """
        F1 score for left boundary matches.
        """
        precision = self.right_match_precision()
        recall = self.right_match_recall()
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def partial_match_precision(self) -> float:
        """
        Precision for left boundary matches.
        """
        if self.actual == 0:
            return 0.0
        return self.partial_match_tp / self.actual
    
    def partial_match_recall(self) -> float:
        """
        Recall for left boundary matches.
        """
        if self.possible == 0:
            return 0.0
        return self.partial_match_tp / self.possible
    
    def partial_match_f1(self) -> float:
        """
        F1 score for left boundary matches.
        """
        precision = self.partial_match_precision()
        recall = self.partial_match_recall()
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)
    
    def partial_credit_ratio(self) -> float:
        total = len(self.overlap_credit_list)
        partial_count = 0
        for mention in self.overlap_credit_list:
            if mention[2] != 1:
                partial_count += 1
        if total == 0:
            return 0.0
        return partial_count/total

    def merge(self, other_scorer: "Scorer") -> None:
        """
        Adds the other Scorer's counts to this one.
        """
        self.true_positives += other_scorer.true_positives
        self.false_positives += other_scorer.false_positives
        self.false_negatives += other_scorer.false_negatives

        self.left_match_tp += other_scorer.left_match_tp
        self.left_match_fp += other_scorer.left_match_fp
        self.left_match_fn += other_scorer.left_match_fn

        self.right_match_tp += other_scorer.right_match_tp
        self.right_match_fp += other_scorer.right_match_fp
        self.right_match_fn += other_scorer.right_match_fn 

        self.partial_match_tp += other_scorer.partial_match_tp
        self.partial_match_fp += other_scorer.partial_match_fp
        self.partial_match_fn += other_scorer.partial_match_fn

        self.overlap_credit_list.extend(other_scorer.overlap_credit_list)
        self.left_credit_list.extend(other_scorer.left_credit_list)
        self.right_credit_list.extend(other_scorer.right_credit_list)

        self.possible += other_scorer.possible
        self.actual += other_scorer.actual

    def print_score_report(self):
        print(f"Exact F1: {self.f1_score() * 100:0.2f}")
        print(f"Left boundary match F1: {self.left_match_f1() * 100:0.2f}")
        print(f"Right boundary match F1: {self.right_match_f1() * 100:0.2f}")
        print(f"Partial boundary match F1: {self.partial_match_f1() * 100:0.2f}")
        print(f"\tPercent given partial credit: {self.partial_credit_ratio() * 100:0.2f}")

    def write_partial_matches(self, path: str, match_type: str = "overlap") -> None:
        """
        Writes matches to CSV with credit column for overlap/left/right.
        """
        with open(path, mode='w', encoding='utf8') as file:
            file.write("gold, prediction, credit\n") 
            if match_type == "overlap":
                credit_list = self.overlap_credit_list
            elif match_type == "left":
                credit_list = self.left_credit_list
            elif match_type == "right":
                credit_list = self.right_credit_list
            else:
                print(f"Warning: Unknown match_type '{match_type}' for writing partial matches.")
                return # Or raise error

            # Write data from the selected list
            for gold, pred, cred in credit_list:
                 # Ensure text fields don't contain commas or handle quoting
                gold_text = f'"{gold.text}"' if ',' in gold.text else gold.text
                pred_text = f'"{pred.text}"' if ',' in pred.text else pred.text
                file.write(f"{gold_text},{pred_text},{cred}\n")

    def get_score_dict(self):
        return {
            'Metric': ['Exact Match', 'Left Boundary', 'Right Boundary', 'Partial (Overlap)'],
            'Precision': [
                self.precision(),
                self.left_match_precision(),
                self.right_match_precision(),
                self.partial_match_precision()
            ],
            'Recall': [
                self.recall(),
                self.left_match_recall(),
                self.right_match_recall(),
                self.partial_match_recall()
            ],
            'F1 Score': [
                self.f1_score(),
                self.left_match_f1(),
                self.right_match_f1(),
                self.partial_match_f1()
            ]
        }

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