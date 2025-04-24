import unittest

from scorer import Mention, Scorer


class TestStrictEvaluation(unittest.TestCase):
    def test_precision(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        predictions = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 3, 5, "said San")]
        scores = Scorer(reference, predictions)
        self.assertAlmostEqual(1/2, scores.precision())

    def test_recall(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        predictions = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 3, 5, "said San")]
        scores = Scorer(reference, predictions)
        self.assertAlmostEqual(1/3, scores.recall())

    def test_f1(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        predictions = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 3, 5, "said San")]
        scores = Scorer(reference, predictions)
        self.assertAlmostEqual(2/5, scores.f1_score())


class TestLeftMatch(unittest.TestCase):
    def test_left_match_precision(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        predictions = [Mention("PER", 0, 1, "Allen"), Mention("ORG", 3, 5, "said San")]
        scores = Scorer(reference, predictions)
        self.assertAlmostEqual(1/2, scores.left_match_precision())

    def test_left_match_recall(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        predictions = [Mention("PER", 0, 1, "Allen"), Mention("ORG", 3, 5, "said San")]
        scores = Scorer(reference, predictions)
        self.assertAlmostEqual(1/3, scores.left_match_recall())

    def test_left_match_f1(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        predictions = [Mention("PER", 0, 1, "Allen"), Mention("ORG", 3, 5, "said San")]
        scores = Scorer(reference, predictions)
        self.assertAlmostEqual(2/5, scores.left_match_f1())


class TestRightMatch(unittest.TestCase):
    def test_right_match_precision(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction = [Mention("PER", 1, 2, "Iverson"), Mention("LOC", 5, 6, "Francisco")]
        scores = Scorer(reference, prediction)
        self.assertAlmostEqual(1, scores.right_match_precision())

    def test_right_match_recall(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction = [Mention("PER", 1, 2, "Iverson"), Mention("LOC", 5, 6, "Francisco")]
        scores = Scorer(reference, prediction)
        self.assertAlmostEqual(2/3, scores.right_match_recall())

    def test_right_match_f1(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction = [Mention("PER", 1, 2, "Iverson"), Mention("LOC", 5, 6, "Francisco")]
        scores = Scorer(reference, prediction)
        self.assertAlmostEqual(4/5, scores.right_match_f1())

class TestPartialBoundaryMatching(unittest.TestCase):
    pass

