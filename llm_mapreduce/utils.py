def chunk_text(text, context_window, tokenizer):
    """
    Splits the text into chunks based on token count to fit within the model's context window.
    
    Parameters:
    - text: The long text to be chunked.
    - context_window: The maximum token length for each chunk.
    - tokenizer: Tokenizer used to calculate token length.
    
    Returns:
    - List of text chunks.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), context_window):
        chunk_tokens = tokens[i:i + context_window]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def group_chunks(chunks, collapse_threshold, tokenizer):
    """
    Groups chunks for the collapse stage if they exceed a token-based threshold.

    Parameters:
    - chunks: List of mapped results (each is a dictionary with "text" field).
    - collapse_threshold: Maximum token length before grouping.
    - tokenizer: The tokenizer associated with the model.

    Returns:
    - List of grouped chunks, each within the collapse threshold.
    """
    grouped = []
    current_group = []
    current_length = 0
    
    for chunk in chunks:
        # Extract the text content of the chunk dictionary for token length calculation
        chunk_text = chunk.get("text", "")
        
        if not chunk_text:
            continue  # Skip chunks with empty text
        
        token_length = len(tokenizer.encode(chunk_text))
        
        # Check if adding the chunk would exceed the collapse threshold
        if current_length + token_length > collapse_threshold:
            grouped.append(current_group)
            current_group = [chunk]
            current_length = token_length
        else:
            current_group.append(chunk)
            current_length += token_length
    
    # Add the last group if not empty
    if current_group:
        grouped.append(current_group)
        
    return grouped

def process_chunk(model, chunk, query, context_window):
    """
    Processes a single chunk using the model.
    
    Parameters:
    - model: The language model.
    - chunk: Text chunk.
    - query: Query for processing.
    - context_window: Maximum token length for each chunk.
    
    Returns:
    - Model's output for the chunk.
    """
    combined_input = f"{query} {chunk}"
    
    # Tokenize with truncation to fit within context window
    inputs = model.tokenizer(
        combined_input, 
        return_tensors="pt", 
        max_length=context_window, 
        truncation=True
    ).to(model.device)
    
    output = model.model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=False,  # Deterministic for summarization
        pad_token_id=model.tokenizer.eos_token_id
    )

    output_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    return {
        "text": output_text,
        "rationale": "Generated based on combine text",
        "answer": output_text
    }

