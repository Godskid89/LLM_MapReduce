import logging
from .structured_info_protocol import StructuredInfoProtocol
from .confidence_calibration import ConfidenceCalibrator
from .utils import chunk_text, group_chunks, process_chunk

class MapReduceLLM:
    def __init__(self, model, context_window=4096, collapse_threshold=2048):
        """
        Initializes the MapReduceLLM with the model and parameters.
        
        Parameters:
        - model: The language model with a `generate` or similar method.
        - context_window: The maximum context window size of the model.
        - collapse_threshold: Threshold for reducing grouped chunks if they exceed context window.
        """
        self.model = model
        self.context_window = context_window
        self.collapse_threshold = collapse_threshold
        self.protocol = StructuredInfoProtocol()
        self.calibrator = ConfidenceCalibrator()

    def chunk_text(self, text, chunk_size):
        """
        Splits the input text into manageable chunks based on token limits.
        """
        tokens = self.model.tokenizer.encode(text)
        return [self.model.tokenizer.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

    def map_stage(self, chunks, query):
        """
        Maps each chunk to a processed result.
        """
        mapped_results = []
        for chunk in chunks:
            result = process_chunk(self.model, chunk, query)
            result.setdefault("text", "")
            mapped_results.append(self.protocol.format_mapped_result(result))
        return mapped_results

    def collapse_stage(self, mapped_results, query):
        """
        Collapses mapped results into fewer chunks, each within the token limit.
        """
        groups = group_chunks(mapped_results, self.collapse_threshold, self.model.tokenizer)
        collapsed_results = []
        for group in groups:
            combined_text = " ".join([r["text"] for r in group])
            
            # Ensure combined text doesn't exceed the model's limit by truncating
            if len(self.model.tokenizer.encode(combined_text)) > self.context_window:
                group = self.chunk_text(combined_text, self.collapse_threshold)
            
            collapsed_result = self.protocol.collapse_results(self.model, group, query)
            collapsed_results.append(collapsed_result)
        return collapsed_results

    def reduce_stage(self, collapsed_results, query):
        """
        Reduces collapsed results to a final answer, ensuring token limits are respected.
        """
        combined_text = " ".join([res["text"] for res in collapsed_results])
        
        # Truncate if combined text exceeds the model's limit
        if len(self.model.tokenizer.encode(combined_text)) > self.context_window:
            chunks = self.chunk_text(combined_text, self.context_window)
            reduced_text = ""
            for chunk in chunks:
                reduced_chunk = self.model.generate(query=query + chunk)["text"]
                reduced_text += " " + reduced_chunk
            combined_text = reduced_text.strip()
        else:
            combined_text = combined_text
        
        # Generate the final answer from the combined text
        final_answer = self.model.generate(query=query + combined_text)
        return final_answer

    def process_long_text(self, document, query):
        """
        Processes a long text document by applying the MapReduce framework.
        
        Parameters:
        - document: The full text to be processed.
        - query: The query or question being asked of the document.
        
        Returns:
        - The final aggregated answer.
        """
        # Automatically chunk the document to fit within the model's context window
        chunks = chunk_text(document, self.context_window, self.model.tokenizer)
        
        # Apply map stage to each chunk
        mapped_results = self.map_stage(chunks, query)
        
        # Collapse mapped results
        collapsed_results = self.collapse_stage(mapped_results, query)
        
        # Apply the reduce stage to get the final result
        final_result = self.reduce_stage(collapsed_results, query)
        if isinstance(final_result, dict):
            return final_result
        else:
            return {"answer": final_result}
