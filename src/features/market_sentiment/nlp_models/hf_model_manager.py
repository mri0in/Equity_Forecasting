# File: src/features/market_sentiment/nlp_models/hf_model_manager.py

"""
Hugging Face Model Manager

Core Responsibilities:
    - Interface with Hugging Face Hub to download and load NLP models.
    - Handle authentication tokens (if required for private models).
    - Cache models locally to avoid repeated downloads.
    - Provide initialized transformers.pipeline objects for inference.
    - Centralize error handling and logging for model operations.

System Design Principle:
    - Manager Pattern → Encapsulates Hugging Face logic separately from business logic.
    - Strict lazy-loading to avoid side effects during app startup.
"""

import os
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
from dotenv import load_dotenv

from src.features.market_sentiment.nlp_models.model_registry import ModelRegistry

if TYPE_CHECKING:
    from transformers import Pipeline

load_dotenv()

class HFModelManager:
    """
    Manages Hugging Face models for sentiment analysis.
    """

    def __init__(self, token_env_var: str = "HF_TOKEN") -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._registry = ModelRegistry()
        self._token_env_var = token_env_var
        self._hf_token: Optional[str] = os.getenv(token_env_var)

        self._pipelines: Dict[str, "Pipeline"] = {}
        self._transformers_loaded = False

        if self._hf_token:
            self._logger.info("Using Hugging Face token from %s", token_env_var)
        else:
            self._logger.warning("No Hugging Face token found (public models only).")

    def _load_transformers(self):
        """
        Lazy-load transformers only when actually needed.
        """
        if self._transformers_loaded:
            return

        try:
            from transformers import (
                pipeline,
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
        except Exception as e:
            self._logger.exception("Failed to import transformers")
            raise RuntimeError("Transformers import failed") from e

        self._pipeline_fn = pipeline
        self._AutoTokenizer = AutoTokenizer
        self._AutoModelForSequenceClassification = AutoModelForSequenceClassification
        self._transformers_loaded = True

    def load_pipeline(self, model_name: str):
        if model_name in self._pipelines:
            return self._pipelines[model_name]

        self._load_transformers()

        model_info: Dict[str, Any] = self._registry.get_model(model_name)

        try:
            tokenizer = self._AutoTokenizer.from_pretrained(
                model_info["hf_id"],
                token=self._hf_token,
            )

            model = self._AutoModelForSequenceClassification.from_pretrained(
                model_info["hf_id"],
                token=self._hf_token,
            )

            nlp_pipeline = self._pipeline_fn(
                task=model_info["task"],
                model=model,
                tokenizer=tokenizer,
            )

            self._pipelines[model_name] = nlp_pipeline
            return nlp_pipeline

        except Exception as e:
            self._logger.exception("HF model load failed: %s", model_name)
            raise RuntimeError(f"Failed to load HF model '{model_name}'") from e

    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        return self._registry.list_models()
