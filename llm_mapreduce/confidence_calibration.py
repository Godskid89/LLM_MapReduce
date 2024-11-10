class ConfidenceCalibrator:
    def calibrate_score(self, rationale):
        """
        Assigns a confidence score based on the rationale text.
        
        Parameters:
        - rationale: Text providing reasoning or support for the answer.
        
        Returns:
        - An integer confidence score (0 to 5).
        """
        rationale = rationale.lower()
        if "no relevant information found" in rationale or "no information" in rationale:
            return 0
        elif "highly supported" in rationale or "high confidence" in rationale:
            return 5
        elif "partially inferred" in rationale or "medium confidence" in rationale:
            return 3
        else:
            return 1  # Default low confidence if rationale doesn't match any keywords
