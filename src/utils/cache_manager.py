# src/utils/cache_manager.py
"""
Centralized cache management with YAML-configured paths.
Supports in-memory LRU caching + disk-based persistence + TTL.
"""

from __future__ import annotations
import os
import joblib
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
from src.utils.logger import get_logger
import yaml

logger = get_logger(__name__)


class CacheManager:
    """
    Centralized cache manager for all modules.
    Loads cache paths from YAML config.
    """

    _instance = None

    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None) -> CacheManager:
        """
        Singleton access to a global CacheManager instance.
        """
        if cls._instance is None:
            cls._instance = cls(config_path or Path("config/config_paths.yaml"))
        return cls._instance

    def __init__(self, config_path: Path, max_mem_items: int = 128) -> None:
        """
        Initialize CacheManager with YAML-configured paths.

        Args:
            config_path (Path): Path to YAML file defining cache directories.
            max_mem_items (int): Max items in in-memory LRU cache.
        """
        self.max_mem_items = max_mem_items
        self._memory_cache = {}
        self._ttl_cache = {}

        # Load cache paths from YAML
        if not config_path.exists():
            raise FileNotFoundError(f"Cache config YAML not found: {config_path}")

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Convert to Path objects
        cache_paths = raw_config["datalake"]["cache"]
        self.CACHE_CONFIG = {k: Path(v) for k, v in cache_paths.items()}

        # Ensure directories exist
        for module, path in self.CACHE_CONFIG.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured cache dir for module '{module}': {path}")

        logger.info(f"CacheManager initialized with LRU size {self.max_mem_items}")

    # ---------------------
    # Key Utilities
    # ---------------------
    def make_key(self, identifier: str, suffix: str) -> str:
        """
        Generate a consistent cache key filename.
        """
        safe_id = identifier.replace(" ", "_").replace("/", "_")
        return f"{safe_id}_{suffix}.joblib"

    def _resolve_path(self, module: str, key: str) -> Path:
        """
        Resolve full disk path for a given module + key.
        """
        if module not in self.CACHE_CONFIG:
            raise ValueError(f"Module '{module}' not in cache config")
        return self.CACHE_CONFIG[module] / key

    # ---------------------
    # Core Cache Operations
    # ---------------------
    def save(self, module: str, key: str, obj: Any, ttl: Optional[int] = None) -> None:
        """
        Save object to memory + disk cache for a specific module.
        """
        path = self._resolve_path(module, key)
        joblib.dump(obj, path)
        logger.info(f"Saved cache to disk: {path} ({round(os.path.getsize(path)/1024, 2)} KB)")

        # In-memory save with simple LRU
        if len(self._memory_cache) >= self.max_mem_items:
            oldest_key = next(iter(self._memory_cache))
            self._memory_cache.pop(oldest_key)
            self._ttl_cache.pop(oldest_key, None)

        self._memory_cache[key] = obj
        if ttl:
            self._ttl_cache[key] = datetime.now() + timedelta(seconds=ttl)

        logger.debug(f"Saved cache in memory: {module}:{key}, ttl={ttl}")

    def exists(self, module: str, key: str) -> bool:
        """
        Check if a cached object exists (memory first, then disk).
        """
        # Memory
        if key in self._memory_cache:
            expiry = self._ttl_cache.get(key)
            if expiry and datetime.now() > expiry:
                self._memory_cache.pop(key)
                self._ttl_cache.pop(key, None)
                logger.info(f"Memory cache expired for key: {module}:{key}")
            else:
                return True

        # Disk
        path = self._resolve_path(module, key)
        return path.exists()

    def load(self, module: str, key: str) -> Any:
        """
        Load cached object (memory first, then disk). Raises FileNotFoundError.
        """
        # Memory
        if key in self._memory_cache:
            expiry = self._ttl_cache.get(key)
            if expiry and datetime.now() > expiry:
                self._memory_cache.pop(key)
                self._ttl_cache.pop(key, None)
                logger.info(f"Memory cache expired for key: {module}:{key}")
            else:
                logger.info(f"Cache hit (memory): {module}:{key}")
                return self._memory_cache[key]

        # Disk
        path = self._resolve_path(module, key)
        if path.exists():
            obj = joblib.load(path)
            self.save(module, key, obj)  # Refresh memory
            logger.info(f"Cache hit (disk): {module}:{key}")
            return obj

        logger.warning(f"Cache miss for key: {module}:{key}")
        raise FileNotFoundError(f"No cached file for key: {module}:{key}")

    def invalidate(self, module: Optional[str] = None, pattern: Optional[str] = None) -> None:
        """
        Invalidate memory + disk cache matching a pattern for a module or all modules.
        """
        modules = [module] if module else list(self.CACHE_CONFIG.keys())
        for mod in modules:
            path = self.CACHE_CONFIG[mod]
            search_pattern = pattern or "*.joblib"
            for f in path.glob(search_pattern):
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

    def list_cache(self, module: Optional[str] = None) -> list[str]:
        """
        List all cached keys (memory + disk) for a module or all modules.
        """
        modules = [module] if module else list(self.CACHE_CONFIG.keys())
        all_keys = []
        for mod in modules:
            disk_files = [f.name for f in self.CACHE_CONFIG[mod].glob("*.joblib")]
            mem_keys = list(self._memory_cache.keys())
            all_keys.extend(disk_files + mem_keys)
        return sorted(set(all_keys))

    def backend_info(self) -> str:
        return f"Memory items: {len(self._memory_cache)}, Modules: {list(self.CACHE_CONFIG.keys())}"
