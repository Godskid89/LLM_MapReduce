import unittest
from llm_mapreduce.structured_info_protocol import StructuredInfoProtocol

class MockModel:
    def generate(self, query):
        return {
            "text": "Mock summary",
            "rationale": "Summarized rationale",
            "answer": "Summary of ecosystem"
        }

class TestStructuredInfoProtocol(unittest.TestCase):
    def setUp(self):
        self.protocol = StructuredInfoProtocol()

    def test_collapse_results(self):
        group = [
            {"extracted_info": "Info about rivers.", "answer": "Rivers help biodiversity."},
            {"extracted_info": "Info about flora.", "answer": "Flora is diverse."}
        ]
        query = "Summarize the ecosystem."
        mock_model = MockModel()
        collapsed_result = self.protocol.collapse_results(mock_model, group, query)
        self.assertIn("answer", collapsed_result)
        self.assertIn("rationale", collapsed_result)

if __name__ == "__main__":
    unittest.main()
