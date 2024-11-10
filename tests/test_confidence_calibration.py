import unittest
from llm_mapreduce.confidence_calibration import ConfidenceCalibrator

class TestConfidenceCalibrator(unittest.TestCase):
    def setUp(self):
        self.calibrator = ConfidenceCalibrator()

    def test_high_confidence(self):
        rationale = "This answer is highly supported by the provided text."
        score = self.calibrator.calibrate_score(rationale)
        self.assertEqual(score, 5)

    def test_medium_confidence(self):
        rationale = "This answer has some basis but is partially inferred."
        score = self.calibrator.calibrate_score(rationale)
        self.assertEqual(score, 3)

    def test_no_information(self):
        rationale = "No relevant information found."
        score = self.calibrator.calibrate_score(rationale)
        self.assertEqual(score, 0)

if __name__ == "__main__":
    unittest.main()
