from abc import ABC, abstractmethod

class AgreementStrategy(ABC):
    """
    Abstract base class for all agreement strategies.
    """
    @abstractmethod
    def check_agreement(self, responses):
        """
        Checks for agreement among a list of responses.

        Args:
            responses: A list of response objects from the LLM clients.

        Returns:
            A boolean indicating whether the responses agree.
        """
        pass


class StrictAgreement(AgreementStrategy):
    """
    Checks for strict, character-for-character agreement among responses.
    """
    def check_agreement(self, responses):
        if not responses:
            return False

        first_response_content = responses[0].choices[0].message.content.strip()

        for response in responses[1:]:
            other_response_content = response.choices[0].message.content.strip()
            if other_response_content != first_response_content:
                print(f"Disagreement found:\n  - '{first_response_content}'\n  - '{other_response_content}'")
                return False
        
        print("Strict agreement found.")
        return True


class SemanticAgreement(AgreementStrategy):
    """
    Checks for semantic agreement using the lightweight FastEmbed library.
    
    This strategy uses a highly optimized sentence-transformer model (bge-small)
    to generate embeddings and check for semantic similarity. It is designed
    to be fast and efficient, even on CPU.
    """
    _model_cache = {}

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", threshold=0.90):
        self.model_name = model_name
        self.threshold = threshold
        self._model = self._get_model()

    def _get_model(self):
        """Lazily loads and caches the FastEmbed model."""
        if self.model_name not in self._model_cache:
            try:
                from fastembed import TextEmbedding
                self._model_cache[self.model_name] = TextEmbedding(model_name=self.model_name)
            except ImportError:
                raise ImportError(
                    "SemanticAgreement requires the 'fastembed' package. "
                    "Please install it with: pip install cascade-inference[semantic]"
                )
        return self._model_cache[self.model_name]

    def check_agreement(self, responses):
        if len(responses) < 2:
            return True

        from scipy.spatial.distance import cosine

        contents = [res.choices[0].message.content for res in responses]
        
        embeddings = list(self._model.embed(contents))
        
        first_embedding = embeddings[0]
        for other_embedding in embeddings[1:]:
            similarity = 1 - cosine(first_embedding, other_embedding)
            
            if similarity < self.threshold:
                print(f"Semantic disagreement found (Similarity: {similarity:.4f} < Threshold: {self.threshold})")
                return False
        
        print(f"Semantic agreement found (All similarities >= {self.threshold})")
        return True 