# File: src/features/market_sentiment/nlp_models/hf_model_manager.py

"""
Hugging Face Model Manager

Core Responsibilities:
    - Interface with Hugging Face Hub to download and load NLP models.
    - Handle authentication tokens (if required for private models).
    - Cache models locally to avoid repeated downloads.
    - Provide initialized `transformers.pipeline` objects for inference.
    - Centralize error handling and logging for model operations.

System Design Principle:
    - Manager Pattern → Encapsulates Hugging Face logic separately from business logic.
    - Decouples model loading from usage (SentimentModel doesn’t worry about pipelines).
"""

import os
import logging
from typing import Optional, Dict, Any

from transformers import pipeline, Pipeline
from src.features.market_sentiment.nlp_models.model_registry import ModelRegistry


class HFModelManager:
    """
    Manages Hugging Face models for sentiment analysis.
    """

    def __init__(self, token_env_var: str = "HF_TOKEN") -> None:
        """
        Initialize the Hugging Face Model Manager.

        Args:
            token_env_var (str): Environment variable storing the HF API token.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._registry = ModelRegistry()
        self._token_env_var = token_env_var
        self._hf_token: Optional[str] = os.getenv(token_env_var)

        if self._hf_token:
            self._logger.info("Using Hugging Face token from environment variable: %s", token_env_var)
        else:
            self._logger.warning("No Hugging Face token found. Public models only.")

        # Cache for loaded pipelines
        self._pipelines: Dict[str, Pipeline] = {}

    def load_pipeline(self, model_name: str) -> Pipeline:
        """
        Load a Hugging Face sentiment pipeline for the specified model.

        Args:
            model_name (str): Name of the model as registered in ModelRegistry.

        Returns:
            Pipeline: A Hugging Face pipeline ready for inference.

        Raises:
            ValueError: If model is not found in registry.
        """
        if model_name in self._pipelines:
            self._logger.debug("Pipeline for '%s' fetched from cache.", model_name)
            return self._pipelines[model_name]

        model_info: Dict[str, Any] = self._registry.get_model(model_name)

        self._logger.info(
            "Loading Hugging Face model '%s' (%s)", model_name, model_info["hf_id"]
        )

        try:
            nlp_pipeline: Pipeline = pipeline(
                task=model_info["task"],
                model=model_info["hf_id"],
                tokenizer=model_info["hf_id"],
                use_auth_token=self._hf_token if self._hf_token else None,
            )
            self._pipelines[model_name] = nlp_pipeline
            self._logger.info("Pipeline for '%s' loaded successfully.", model_name)
            return nlp_pipeline

        except Exception as e:
            self._logger.exception("Failed to load model '%s': %s", model_name, str(e))
            raise RuntimeError(f"Could not load Hugging Face model '{model_name}'.")

    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered models with their metadata.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of model names to metadata.
        """
        return self._registry.list_models()
