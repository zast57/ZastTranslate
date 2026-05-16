from abc import ABC, abstractmethod

class LLMBackend(ABC):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend to display in UI."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> dict:
        """
        Returns a dictionary of capabilities:
        - languages: list of ISO 639-1 language codes supported (or 'all')
        - vram_gb: approx VRAM needed
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend's required packages are installed."""
        pass

    @abstractmethod
    def load(self):
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model and free memory."""
        pass

    @abstractmethod
    def generate(self, messages: list, max_new_tokens: int = 4096, multiline: bool = False) -> str:
        """Generate text from a list of conversational messages."""
        pass

    def generate_batch(self, messages_list: list, max_new_tokens_list: list, **kwargs) -> list:
        """Batched inference. Default: sequential fallback. Override for real batching."""
        return [
            self.generate(msgs, max_new_tokens=mnt, **kwargs)
            for msgs, mnt in zip(messages_list, max_new_tokens_list)
        ]
