"""
Centralized model manager for MeVe framework.
Implements singleton pattern to avoid multiple model instantiations.
"""

from typing import Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from meve.utils import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Singleton class to manage all ML models used in MeVe pipeline."""

    _instance: Optional["ModelManager"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize models lazily on first access."""
        if not ModelManager._initialized:
            self._sentence_transformer: Optional[SentenceTransformer] = None
            self._cross_encoder: Optional[CrossEncoder] = None
            self._tokenizer: Optional[AutoTokenizer] = None
            self._embedding_model_name: str = "text-embedding-3-small"
            ModelManager._initialized = True
            logger.debug("ModelManager initialized")

    def set_embedding_model(self, model_name: str) -> None:
        """Set the embedding model to use."""
        if model_name != self._embedding_model_name:
            logger.info(f"Changing embedding model from {self._embedding_model_name} to {model_name}")
            self._embedding_model_name = model_name
            self._sentence_transformer = None  # Clear cached model

    @property
    def sentence_transformer(self) -> SentenceTransformer:
        """Get or initialize the sentence transformer model."""
        if self._sentence_transformer is None:
            logger.info(f"Loading SentenceTransformer model: {self._embedding_model_name}")
            self._sentence_transformer = SentenceTransformer(self._embedding_model_name)
        return self._sentence_transformer

    @property
    def cross_encoder(self) -> CrossEncoder:
        """Get or initialize the cross-encoder model."""
        if self._cross_encoder is None:
            logger.info("Loading CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder

    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
        """Get or initialize the GPT-2 tokenizer."""
        if self._tokenizer is None:
            try:
                logger.info("Loading GPT-2 tokenizer")
                self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception as e:
                logger.warning(f"Could not load GPT-2 tokenizer: {e}. Using fallback tokenization.")
                self._tokenizer = None
        return self._tokenizer

    def clear_models(self):
        """Clear all loaded models to free memory."""
        logger.info("Clearing all loaded models")
        self._sentence_transformer = None
        self._cross_encoder = None
        self._tokenizer = None

    @classmethod
    def reset(cls):
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
        cls._initialized = False


# Global accessor functions for backward compatibility
def get_sentence_transformer(model_name: Optional[str] = None) -> SentenceTransformer:
    """Get the shared sentence transformer model."""
    manager = ModelManager()
    if model_name is not None:
        manager.set_embedding_model(model_name)
    return manager.sentence_transformer


def get_cross_encoder() -> CrossEncoder:
    """Get the shared cross-encoder model."""
    return ModelManager().cross_encoder


def get_tokenizer() -> Optional[AutoTokenizer]:
    """Get the shared tokenizer model."""
    return ModelManager().tokenizer
