from flair.data import Label
from typing_extensions import NamedTuple
from typing import Sequence

# this is from COSI 216A HW1
class Mention(NamedTuple):
    """
    Start index inclusive, end index exclusive.
    """
    entity_type: str
    start: int
    end: int

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

    def merge(self, other_scorer: "Scorer") -> None:
        """
        Adds the other Scorer's counts to this one.
        """
        # TODO bad practice to modify these directly?
        self.true_positives += other_scorer.true_positives
        self.false_positives += other_scorer.false_positives
        self.false_negatives += other_scorer.false_negatives


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
            mentions.append(Mention(entity_type, start, end))
        return mentions