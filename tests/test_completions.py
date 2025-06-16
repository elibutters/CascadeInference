import pytest
from cascade.chat import completions

# Mock client classes for testing
class MockClient:
    def __init__(self, name="mock"):
        self.name = name

    def __repr__(self):
        return f"MockClient(name='{self.name}')"

    @property
    def __class__(self):
        return MockClient

def test_create_completion():
    """
    Tests the basic functionality of the create completion function.
    """
    level1_clients = [
        (MockClient("phi3"), "accounts/fireworks/models/phi-3-mini-128k-instruct"),
        (MockClient("gemma"), "gemma-7b-it")
    ]
    level2_client = (MockClient("claude"), "claude-3-5-sonnet-20240620")

    messages = [
        {"role": "user", "content": "What are the key differences between HBM3e and GDDR7 memory?"}
    ]

    response = completions.create(
        level1_clients=level1_clients,
        level2_client=level2_client,
        agreement_strategy="semantic",
        messages=messages,
        temperature=0.5
    )

    assert response is not None
    assert hasattr(response, 'choices')
    assert len(response.choices) == 1
    assert hasattr(response.choices[0], 'message')
    assert hasattr(response.choices[0].message, 'content')
    assert isinstance(response.choices[0].message.content, str)
    assert response.choices[0].message.content == "This is a mock response from Cascade Inference." 