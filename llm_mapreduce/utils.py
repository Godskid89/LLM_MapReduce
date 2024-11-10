def chunk_text(text, context_window, tokenizer):
    """
    Splits the text into chunks fitting within the model's token-based context window.

    Parameters:
    - text: The long text to be chunked.
    - context_window: The maximum token length for each chunk.
    - tokenizer: The tokenizer associated with the model.

    Returns:
    - List of text chunks that fit within the context window.
    """
    tokens = tokenizer.encode(text)
    chunks = [tokenizer.decode(tokens[i:i + context_window]) for i in range(0, len(tokens), context_window)]
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

def process_chunk(model, chunk, query):
    """
    Processes a single chunk using the model, ensuring combined length is within token limits.
    
    Parameters:
    - model: The language model.
    - chunk: Text chunk.
    - query: Query for processing.
    
    Returns:
    - Model's output for the chunk in a structured format.
    """
    combined_input = f"{query}\n{chunk}"
    max_length = model.tokenizer.model_max_length or 1024  # Get max length or default to 1024

    # Tokenize and truncate combined input if necessary
    inputs = model.tokenizer(
        combined_input, return_tensors="pt", max_length=max_length, truncation=True
    ).to(model.device)
    
    # Generate response with a limited max length to avoid exceeding the model's capacity
    output = model.model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=100,  
        temperature=0.7,
        do_sample=True,
        pad_token_id=model.tokenizer.eos_token_id
    )
    
    # Decode and return structured response
    output_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    return {
        "text": output_text,
        "rationale": "Generated response based on combined input.",
        "answer": output_text
        # "answer": output_text.split("\n")[0]  # Use first line for the answer
    }
