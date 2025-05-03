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


class TestPartialMatch(unittest.TestCase):
    def test_no_double_match(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction = [Mention("PER", 1, 3, "Iverson Meta")]
        # this should score as 0.5 partial match tp
        scores = Scorer(reference, prediction)
        self.assertEqual(scores.partial_match_tp, 0.5)
    
    def test_partial_match_precision(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction = [Mention("PER", 0, 2, "Allen Iverson"), Mention("LOC", 5, 6, "Francisco")]
        scores = Scorer(reference, prediction)
        self.assertAlmostEqual(0.75, scores.partial_match_precision())

    def test_partial_match_recall(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction = [Mention("PER", 0, 2, "Iverson"), Mention("LOC", 5, 6, "Francisco")]
        scores = Scorer(reference, prediction)
        self.assertAlmostEqual(0.5, scores.partial_match_recall())

    def test_partial_match_f1(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction = [Mention("PER", 0, 2, "Allen Iverson"), Mention("LOC", 5, 6, "Francisco")]
        scores = Scorer(reference, prediction)
        self.assertAlmostEqual(0.6, scores.partial_match_f1())
        scores.write_partial_matches("predictions/partial")


class TestOverlap(unittest.TestCase):
    # TODO
    pass


class TestMerge(unittest.TestCase):
    def test_merge_exact_match(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 3, 5, "said San")]
        scores = Scorer(reference, prediction)
        reference_two = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction_two = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta")]
        other = Scorer(reference_two, prediction_two)
        scores.merge(other)
        self.assertEqual(3, scores.true_positives)
        self.assertEqual(1, scores.false_positives)
        self.assertEqual(3, scores.false_negatives)

        self.assertAlmostEqual(0.6, scores.f1_score())

    def test_merge_partial_match(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 3, 5, "said San")]
        scores = Scorer(reference, prediction)
        self.assertAlmostEqual(1.5, scores.partial_match_tp)
        reference_two = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        prediction_two = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta")]
        other = Scorer(reference_two, prediction_two)
        scores.merge(other)
        self.assertAlmostEqual(7/8, scores.partial_match_precision())
        self.assertAlmostEqual(3.5/6, scores.partial_match_recall())
        self.assertAlmostEqual(7/10, scores.partial_match_f1())
    
    def test_merge_starting_empty(self) -> None:
        reference = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 2, 3, "Meta"), Mention("LOC", 4, 6, "San Francisco")]
        predictions = [Mention("PER", 0, 2, "Allen Iverson"), Mention("ORG", 3, 5, "said San")]
        scores = Scorer([], [])
        scores.merge(Scorer(reference, predictions))
        self.assertAlmostEqual(2/5, scores.f1_score())