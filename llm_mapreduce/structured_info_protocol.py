from .confidence_calibration import ConfidenceCalibrator

class StructuredInfoProtocol:
    def __init__(self, context_window=1024):
        self.calibrator = ConfidenceCalibrator()
        self.context_window = context_window  # Maximum allowable token length for the model

    def format_mapped_result(self, result):
        """
        Formats the mapped result into the structured protocol.
        
        Parameters:
        - result: The model's response for a single chunk.
        
        Returns:
        - A dictionary containing structured information.
        """
        return {
            "extracted_info": result.get("text", ""),
            "rationale": result.get("rationale", ""),
            "answer": result.get("answer", "NO INFORMATION"),
            "confidence_score": self.calibrator.calibrate_score(result.get("rationale", ""))
        }

    def chunk_text(self, text, model):
        """
        Splits text into chunks that fit within the model's token limit.
        """
        tokens = model.tokenizer.encode(text)
        return [model.tokenizer.decode(tokens[i:i + self.context_window]) for i in range(0, len(tokens), self.context_window)]

    def collapse_results(self, model, group, query):
        """
        Collapses a group of mapped results to a single representation.
        
        Parameters:
        - model: The language model.
        - group: List of mapped results to collapse.
        - query: User query to refocus the collapse process.
        
        Returns:
        - Collapsed result in structured format.
        """
        combined_text = " ".join(item["extracted_info"] for item in group)
        
        # Ensure combined text doesn't exceed the model's token limit by chunking if necessary
        if len(model.tokenizer.encode(combined_text)) > self.context_window:
            chunks = self.chunk_text(combined_text, model)
            collapsed_text = ""
            for chunk in chunks:
                result = model.generate(query=query + chunk)
                collapsed_text += result.get("text", "") + " "
            combined_text = collapsed_text.strip()
        else:
            result = model.generate(query=query + combined_text)

        return self.format_mapped_result(result)

    def reduce_results(self, model, collapsed_results, query):
        """
        Aggregates all collapsed results to form the final answer.
        
        Parameters:
        - model: The language model.
        - collapsed_results: List of collapsed results.
        - query: User query.
        
        Returns:
        - Final aggregated answer.
        """
        combined_text = " ".join(item["extracted_info"] for item in collapsed_results)
        
        # Ensure combined text does not exceed model's token limit
        if len(model.tokenizer.encode(combined_text)) > self.context_window:
            chunks = self.chunk_text(combined_text, model)
            reduced_text = ""
            for chunk in chunks:
                result = model.generate(query=query + chunk)
                reduced_text += result.get("text", "") + " "
            final_text = reduced_text.strip()
        else:
            result = model.generate(query=query + combined_text)
            final_text = result.get("text", "")
        
        return {"answer": final_text}
