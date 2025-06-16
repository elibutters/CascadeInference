# Cascade Inference

Cascade based inference for large language models.

## Installation

```bash
pip install cascade-inference

# To use semantic agreement, install the optional dependencies:
pip install cascade-inference[semantic]
```

## Basic Usage

Using the library is as simple as a standard OpenAI API call.

```python
from openai import OpenAI
import cascade
import os

# Setup your clients
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Call the create function directly
response = cascade.chat.completions.create(
    # Provide the ensemble of fast clients
    level1_clients=[
        (client, "meta-llama/llama-3.1-8b-instruct"),
        (client, "google/gemini-flash-1.5")
    ],
    # Provide the single, powerful client for escalation
    level2_client=(client, "openai/gpt-4o"),
    # Define the comparison strategy on the fly
    agreement_strategy="semantic", # or "strict"
    # Pass in the standard OpenAI-style arguments
    messages=[
        {"role": "user", "content": "What are the key differences between HBM3e and GDDR7 memory?"}
    ]
)

# The response object looks just like a standard OpenAI response
print(response.choices[0].message.content)
```

## Advanced Configuration

For more control, you can pass a dictionary to the `agreement_strategy` parameter. This allows you to fine-tune the agreement logic.

### 1. Changing the Semantic Similarity Threshold

You can adjust how strictly the semantic comparison is applied. The `threshold` is a value between 0 and 1, where 1 is a perfect match. The default is `0.9`.

```python
response = cascade.chat.completions.create(
    # ... clients and messages ...
    agreement_strategy={
        "name": "semantic",
        "threshold": 0.95  # Require a 95% similarity match
    },
    # ...
)
```

### 2. Using a Different Embedding Model

The default model is `BAAI/bge-small-en-v1.5`, which is fast and lightweight. You can specify any other model compatible with the [FastEmbed](https://github.com/qdrant/fastembed) library.

The library will automatically download and cache the new model on the first run.

```python
response = cascade.chat.completions.create(
    # ... clients and messages ...
    agreement_strategy={
        "name": "semantic",
        "model_name": "BAAI/bge-base-en-v1.5", # A larger, more powerful model
        "threshold": 0.99 # It's good practice to adjust the threshold for a new model
    },
    # ...
)
``` 