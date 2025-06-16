# Cascade Inference

Cascade based inference for large language models.

## Installation

```bash
pip install cascade-inference
```

## Usage

```python
from openai import OpenAI
from anthropic import Anthropic
import cascade

# Level 1 Clients (The cheap, fast ensemble)
client_phi3 = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key="...") 
client_gemma = OpenAI(base_url="https://api.groq.com/openai/v1", api_key="...")

# Level 2 Client (The powerful, expensive verifier)
client_claude = Anthropic(api_key="sk-...")

response = arbiter.chat.completions.create(
    # Provide the ensemble of fast clients
    level1_clients=[
        (client_phi3, "accounts/fireworks/models/phi-3-mini-128k-instruct"),
        (client_gemma, "gemma-7b-it")
    ],
    
    # Provide the single, powerful client for escalation
    level2_client=(client_claude, "claude-3-5-sonnet-20240620"),
    
    # Define the comparison strategy on the fly
    agreement_strategy="semantic", # or "strict"
    
    # Pass in the standard OpenAI-style arguments
    messages=[
        {"role": "user", "content": "What are the key differences between HBM3e and GDDR7 memory?"}
    ],
    temperature=0.5
)

# The response object looks just like a standard OpenAI response
print(response.choices[0].message.content)
``` 