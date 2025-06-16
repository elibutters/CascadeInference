# Cascade Inference

Cascade based inference for large language models.

## Installation

```bash
pip install cascade-inference
```

## Usage

```python
from openai import OpenAI
import cascade, os

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

response = cascade.chat.completions.create(
    # Provide the ensemble of fast clients
    level1_clients = [
        (client, "meta-llama/llama-3.1-8b-instruct"),
        (client, "meta-llama/llama-3.2-3b-instruct")
    ]
    # Provide the single, powerful client for escalation
    level2_client = (client, "openai/gpt-4o")
    agreement_strategy="semantic", # or "strict"
    messages=[
        {"role": "user", "content": "What are the key differences between HBM3e and GDDR7 memory?"}
    ],
    temperature=0.5
)

# The response object looks just like a standard OpenAI response
print(response.choices[0].message.content)
``` 