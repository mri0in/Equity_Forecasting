#src/data/cache_manager.py
"""
cache_manager.py
================
Centralized cache management module for the Equity Forecasting project.

This module provides the `CacheManager` class, which handles caching of data
throughout the pipeline. It supports both in-memory LRU caching and disk-based caching
with TTL (time-to-live) support.

Features:
- In-memory LRU cache
- Disk-based persistent cache
- TTL (expiration) support
- Logging, type hints, modular design
"""

from __future__ import annotations
import os
import joblib
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
from src.utils.logger import get_logger

from functools import lru_cache


logger = get_logger(__name__)


class CacheManager:
    """
    Dual caching system with in-memory LRU and disk-based persistent cache.
    """

    def __init__(self, base_dir: Path, max_mem_items: int = 128) -> None:
        """
        Initialize the cache manager.

        Parameters
        ----------
        base_dir : Path
            Root directory for disk cache storage.
        max_mem_items : int
            Maximum number of items to store in in-memory LRU cache.
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_mem_items = max_mem_items
        self._memory_cache = {}
        self._ttl_cache = {}

        logger.info(f"CacheManager initialized at {self.base_dir} with LRU size {self.max_mem_items}")

    # -----------------------------------------------------------------
    # Key Utilities
    # -----------------------------------------------------------------
    def make_key(self, identifier: str, suffix: str) -> str:
        """
        Create a consistent cache key.

        Parameters
        ----------
        identifier : str
            Unique identifier (ticker, date, etc.)
        suffix : str
            Type of cached object ('api', 'preprocessed', 'model', etc.)

        Returns
        -------
        str
        """
        safe_id = identifier.replace(" ", "_").replace("/", "_")
        return f"{safe_id}_{suffix}.joblib"

    def _resolve_path(self, key: str) -> Path:
        """
        Resolve full path for a cache key.
        """
        return self.base_dir / key

    # -----------------------------------------------------------------
    # Core Cache Operations
    # -----------------------------------------------------------------
    def save(self, key: str, obj: Any, ttl: Optional[int] = None) -> None:
        """
        Save object to memory + disk cache.

        Parameters
        ----------
        key : str
        obj : Any
        ttl : Optional[int]
            Time-to-live in seconds for in-memory cache.
        """
        # Disk save
        path = self._resolve_path(key)
        joblib.dump(obj, path)
        logger.info(f"Saved cache to disk: {path} ({round(os.path.getsize(path)/1024, 2)} KB)")

        # In-memory save
        if len(self._memory_cache) >= self.max_mem_items:
            # Simple LRU eviction
            oldest_key = next(iter(self._memory_cache))
            self._memory_cache.pop(oldest_key)
            self._ttl_cache.pop(oldest_key, None)

        self._memory_cache[key] = obj
        if ttl:
            self._ttl_cache[key] = datetime.now() + timedelta(seconds=ttl)

        logger.debug(f"Saved cache in memory: {key}, ttl={ttl}")

    def exists(self, key: str) -> bool:
        """
        Check if cache exists (memory first, then disk).
        """
        # Check memory with TTL
        if key in self._memory_cache:
            expiry = self._ttl_cache.get(key)
            if expiry and datetime.now() > expiry:
                # Expired
                self._memory_cache.pop(key)
                self._ttl_cache.pop(key, None)
                logger.info(f"Memory cache expired for key: {key}")
            else:
                return True

        # Check disk
        path = self._resolve_path(key)
        return path.exists()

    def load(self, key: str) -> Any:
        """
        Load cached object (memory first, then disk).

        Raises
        ------
        FileNotFoundError
        """
        # Memory load
        if key in self._memory_cache:
            expiry = self._ttl_cache.get(key)
            if expiry and datetime.now() > expiry:
                # Expired
                self._memory_cache.pop(key)
                self._ttl_cache.pop(key, None)
                logger.info(f"Memory cache expired for key: {key}")
            else:
                logger.info(f"Cache hit (memory): {key}")
                return self._memory_cache[key]

        # Disk load
        path = self._resolve_path(key)
        if path.exists():
            obj = joblib.load(path)
            # Refresh memory cache
            self.save(key, obj)
            logger.info(f"Cache hit (disk): {key}")
            return obj

        logger.warning(f"Cache miss for key: {key}")
        raise FileNotFoundError(f"No cached file for key: {key}")

    def invalidate(self, pattern: Optional[str] = None) -> None:
        """
        Invalidate memory + disk cache matching pattern.
        """
        search_pattern = pattern if pattern else "*.joblib"

        # Disk
        for f in self.base_dir.glob(search_pattern):
            try:
                f.unlink()
                logger.info(f"Deleted disk cache: {f}")
            except Exception as e:
                logger.error(f"Failed to delete {f}: {e}")

        # Memory
        keys_to_delete = [k for k in self._memory_cache if pattern is None or pattern in k]
        for k in keys_to_delete:
            self._memory_cache.pop(k, None)
            self._ttl_cache.pop(k, None)
            logger.info(f"Deleted memory cache: {k}")

    def list_cache(self) -> list[str]:
        """
        List all cached keys (memory + disk)
        """
        disk_files = [f.name for f in self.base_dir.glob("*.joblib")]
        mem_keys = list(self._memory_cache.keys())
        return sorted(set(disk_files + mem_keys))

    def backend_info(self) -> str:
        return f"Memory items: {len(self._memory_cache)}, Disk dir: {self.base_dir}"
