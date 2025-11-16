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
            ModelManager._initialized = True
            logger.debug("ModelManager initialized")

    @property
    def sentence_transformer(self) -> SentenceTransformer:
        """Get or initialize the sentence transformer model."""
        if self._sentence_transformer is None:
            logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
            self._sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
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
def get_sentence_transformer() -> SentenceTransformer:
    """Get the shared sentence transformer model."""
    return ModelManager().sentence_transformer


def get_cross_encoder() -> CrossEncoder:
    """Get the shared cross-encoder model."""
    return ModelManager().cross_encoder


def get_tokenizer() -> Optional[AutoTokenizer]:
    """Get the shared tokenizer model."""
    return ModelManager().tokenizer
