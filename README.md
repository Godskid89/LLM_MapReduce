# **LLM_MapReduce**

`llm_mapreduce` is an open-source Python package that enables Large Language Models (LLMs) to process long documents efficiently by implementing a MapReduce-inspired framework. This package lets you extend the capabilities of any LLM to handle long texts without retraining the model. It works by dividing documents into manageable chunks, processing them independently, and then aggregating the results to produce a coherent answer.

## **Overview**

Many LLMs are limited by a fixed context window, making it difficult to process extended texts in a single pass. `llm_mapreduce` overcomes this limitation using a **three-stage framework** inspired by MapReduce:
1. **Map Stage**: The document is split into chunks, each processed by the model to extract relevant information.
2. **Collapse Stage**: The mapped results are grouped and summarized, keeping them within the model’s context window.
3. **Reduce Stage**: The results from the collapse stage are aggregated to provide a final answer, resolving inter-chunk dependencies and conflicts.

## **Features**

- **Model-Agnostic**: Works with any LLM, including OpenAI's GPT, Hugging Face models, and others.
- **Training-Free**: No need to fine-tune or retrain the model.
- **Extends Context Window**: Supports long-document processing by dividing, summarizing, and aggregating content.
- **Structured Information Protocol**: Organizes intermediate outputs into a structured format, ensuring coherence across chunks.
- **In-Context Confidence Calibration**: Assigns confidence scores to intermediate results for accurate conflict resolution.

## **Installation**

```bash
pip install llm_mapreduce
```

## **Usage**

### **Quick Start with OpenAI GPT**

To use `llm_mapreduce` with OpenAI's GPT models, you need an API key. Set up an OpenAI model wrapper and initialize `MapReduceLLM` to process a large document.

#### 1. **Set up OpenAI API Key**

```bash
export OPENAI_API_KEY='your-openai-api-key'
```

#### 2. **Code Example**

```python
import openai
from llm_mapreduce.mapreduce import MapReduceLLM

# Initialize OpenAI API
openai.api_key = "your-openai-api-key"

class OpenAIModelWrapper:
    """Wrapper to make OpenAI API compatible with MapReduceLLM."""
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name

    def generate(self, query):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": query}],
            max_tokens=500,
        )
        output_text = response.choices[0].message['content']
        return {
            "text": output_text,
            "rationale": output_text,
            "answer": output_text.split("\n")[0]  # Simple answer parsing
        }

# Initialize the wrapper and MapReduceLLM
model = OpenAIModelWrapper(model_name="gpt-4")
mapreduce_llm = MapReduceLLM(model=model, context_window=4096)

# Define the document and query
document = """Your large document text goes here..."""
query = "Summarize the key points."

# Process the document
result = mapreduce_llm.process_long_text(document, query)
print("Final Result:", result)
```

### **Configuring MapReduceLLM**

- **`context_window`**: Define the maximum chunk size based on the model’s token limit.
- **`collapse_threshold`**: Controls when chunks should be grouped and summarized in the Collapse stage.

## **Components**

### `MapReduceLLM` Class

This is the main class that implements the MapReduce process for long text handling.

#### Methods:
- **`map_stage()`**: Processes each chunk with the model.
- **`collapse_stage()`**: Summarizes mapped results when they exceed the context window.
- **`reduce_stage()`**: Aggregates collapsed results to generate the final output.

### `StructuredInfoProtocol`

Formats intermediate outputs for each chunk into a structured format with:
- **Extracted Information**: Key data relevant to the query.
- **Rationale**: Explanation of the answer based on the chunk.
- **Answer**: Intermediate answer based on extracted information.
- **Confidence Score**: Reliability of the answer to manage conflicts between chunks.

### `ConfidenceCalibrator`

Assigns a confidence score to intermediate results based on the rationale, helping resolve conflicts in the reduce stage.

## **Example Applications**

- **Legal and Financial Analysis**: Analyze long legal documents or financial reports to extract critical insights.
- **Scientific Research**: Summarize and query large research papers or datasets.
- **Customer Support**: Summarize and analyze long histories of customer interactions.

## **Development and Contribution**

Contributions are welcome! To set up a development environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm_mapreduce.git
   cd llm_mapreduce
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest
   ```

### **How to Contribute**

- Fork the repository and create a new branch for your feature.
- Submit a pull request with a clear description of your changes.

## **References**

- Zhou, Z., Li, C., Chen, X., Wang, S., Chao, Y., et al. (2024). **LLM×MapReduce: Simplified Long-Sequence Processing Using Large Language Models**. arXiv preprint arXiv:2410.09342.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
