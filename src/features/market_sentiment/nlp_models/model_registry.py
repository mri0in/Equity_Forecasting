# File: src/features/market_sentiment/nlp_models/model_registry.py

"""
Model Registry for available (Hugging Face) NLP Models.

Core Responsibilities:
    - Maintain a registry (dictionary) of available Hugging Face models for sentiment analysis.
    - Provide metadata such as model identifier, description, and usage notes.
    - Allow lookup by key to dynamically select models at runtime.
    - Enforce strict validation: only registered models can be used by SentimentModel.

Design Notes:
    - Decouples model selection from implementation.
    - Enables extendability: add new models without changing sentiment logic.
    - Keeps system clean by centralizing model references.

System Design Principle:
    - Registry Pattern → Provides a single source of truth for available NLP models.
"""

import logging
from typing import Dict, Any


class ModelRegistry:
    """
    Registry for supported Hugging Face sentiment models.
    """

    def __init__(self) -> None:
        """
        Initialize the registry with pre-defined models.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._models: Dict[str, Dict[str, Any]] = {
            "finbert": {
                "hf_id": "ProsusAI/finbert",
                "description": "FinBERT - BERT-based model fine-tuned for financial sentiment analysis.",
                "task": "text-classification",
            },
            "distilbert-sst2": {
                "hf_id": "distilbert-base-uncased-finetuned-sst-2-english",
                "description": "DistilBERT sentiment classifier trained on SST-2 dataset.",
                "task": "text-classification",
            },
            "roberta-financial-news": {
                "hf_id": "yiyanghkust/roberta-financial-news-sentiment",
                "description": "RoBERTa model fine-tuned on financial news sentiment classification.",
                "task": "text-classification",
            },
        }
        self._logger.info("Model registry initialized with %d models.", len(self._models))

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary of all registered models.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of model names to metadata.
        """
        return self._models

    def get_model(self, name: str) -> Dict[str, Any]:
        """
        Retrieve model metadata by name.

        Args:
            name (str): Name of the model (e.g., 'finbert').

        Returns:
            Dict[str, Any]: Model metadata including HF ID and description.

        Raises:
            ValueError: If the model name is not registered.
        """
        if name not in self._models:
            self._logger.error("Attempted to access unregistered model: %s", name)
            raise ValueError(f"Model '{name}' is not registered in the registry.")

        self._logger.debug("Retrieved model info for: %s", name)
        return self._models[name]
