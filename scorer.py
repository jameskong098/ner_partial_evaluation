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

        # TODO consider a different way to store the counts: dictionary?

        # strict evaluation
        self.true_positives = len(reference_set & predictions_set)
        self.false_positives = len(predictions_set - reference_set)
        self.false_negatives = len(reference_set - predictions_set)
        
        # Store raw mentions for partial matching
        self.reference = list(reference)
        self.predictions = list(predictions)

        self.possible = len(self.reference) # used for recall
        self.actual = len(self.predictions) # used for precision

        # self.left_match_tp = 0
        # self.right_match_tp = 0
        # self.partial_match_tp = 0
        # self.type_match_tp = 0
        self.overlap_scores = 0.0
        
        # self._compute_partial_matches()

        left_reference_set = set([(mention.start) for mention in reference])
        left_predictions_set = set([(mention.start) for mention in predictions])
        self.left_match_tp = len(left_reference_set & left_predictions_set)
        self.left_match_fp = len(left_predictions_set - left_reference_set)
        self.left_match_fn = len(left_reference_set - left_predictions_set)

        right_reference_set = set([(mention.end) for mention in reference])
        right_predictions_set = set([(mention.end) for mention in predictions])
        self.right_match_tp = len(right_reference_set & right_predictions_set)
        self.right_match_fp = len(right_predictions_set - right_reference_set)
        self.right_match_fn = len(right_reference_set - right_predictions_set)

        self.partial_match_tp, self.partial_match_fp, self.partial_match_fn, self.partial_credit_dict = self.__count_partial_matches()

        self.overlap_credict_dict = {}
        
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
                # if pred.entity_type != ref.entity_type:
                #     continue
                    
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
    
    def __count_partial_matches(self) -> tuple[int, int, int]:
        partial_match_tp = 0
        partial_match_fp = 0
        partial_match_fn = 0
        partial_credit_dict = {}

        unmatched_references = list(self.reference)
        matched_predictions = set()
        for prediction in self.predictions:
            i = 0
            match_found = False
            while i < len(unmatched_references) and not match_found: # first match approach
                reference = unmatched_references[i]
                if self.__has_overlap(prediction, reference):
                    if prediction.start == reference.start and prediction.end == reference.end:
                        credit = 1
                    else:
                        credit = 0.5 # overlap w/o exact boundary match = 0.5 credit
                    partial_match_tp += credit

                    # TODO there is prob better way to do this
                    partial_credit_dict[reference] = (prediction, credit)

                    unmatched_references.pop(i)
                    matched_predictions.add(prediction)
                    match_found = True
                i += 1
        partial_match_fp = len(set(self.predictions) - matched_predictions)
        partial_match_fn = len(unmatched_references)

        return partial_match_tp, partial_match_fp, partial_match_fn, partial_credit_dict

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
        return self.left_match_tp / (self.left_match_tp + self.left_match_fp)
    
    def left_match_recall(self) -> float:
        """
        Recall for left boundary matches.
        """
        if self.left_match_tp + self.left_match_fn == 0:
            return 0.0
        return self.left_match_tp / (self.left_match_tp + self.left_match_fn)
    
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
        return self.right_match_tp / (self.right_match_tp + self.right_match_fp)
    
    def right_match_recall(self) -> float:
        """
        Recall for left boundary matches.
        """
        if self.right_match_tp + self.right_match_fn == 0:
            return 0.0
        return self.right_match_tp / (self.right_match_tp + self.right_match_fn)
    
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
        total = len(self.partial_credit_dict)
        partial_count = 0
        for mention in self.partial_credit_dict:
            if self.partial_credit_dict[mention][1] != 1:
                partial_count += 1
        if total == 0:
            return 0.0
        return partial_count/total
    
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
        # self.left_match_tp += other_scorer.left_match_tp
        # self.right_match_tp += other_scorer.right_match_tp
        # self.partial_match_tp += other_scorer.partial_match_tp
        # self.type_match_tp += other_scorer.type_match_tp
        self.overlap_scores += other_scorer.overlap_scores
        self.left_match_tp += other_scorer.left_match_tp
        self.left_match_fp += other_scorer.left_match_fp
        self.left_match_fn += other_scorer.left_match_fn

        self.right_match_tp += other_scorer.right_match_tp
        self.right_match_fp += other_scorer.right_match_fp
        self.right_match_fn += other_scorer.left_match_fn

        self.partial_match_tp += other_scorer.partial_match_tp
        self.partial_match_fp += other_scorer.partial_match_fp
        self.partial_match_fn += other_scorer.partial_match_fn
        
        # Append the raw mentions for potential recomputation
        self.reference.extend(other_scorer.reference)
        self.predictions.extend(other_scorer.predictions)

    def print_score_report(self):
        print(f"Exact F1: {self.f1_score() * 100:0.2f}")
        print(f"Left match F1: {self.left_match_f1() * 100:0.2f}")
        print(f"Right match F1: {self.right_match_f1() * 100:0.2f}")
        print(f"Partial match F1: {self.partial_match_f1() * 100:0.2f}")
        print(f"\tPercent given partial credit: {self.partial_credit_ratio() * 100:0.2f}")

    def write_partial_matches(self, path: str) -> None:
        with open(path, mode='w', encoding='utf8') as file:
            file.write("gold, prediction, credit\n")
            for reference in self.partial_credit_dict:
                prediction, credit = self.partial_credit_dict[reference]
                file.write(reference.text + "," + prediction.text + "," + str(credit) + "\n")

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