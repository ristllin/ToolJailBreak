"""Response caching for deduplication across runs."""

import json
from pathlib import Path
from typing import Any

import xxhash

from toolinject.core.schemas import ModelRequest, ModelResponse


class ResponseCache:
    """Content-addressed cache for model responses."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
    
    def _compute_key(self, provider: str, model: str, request: ModelRequest) -> str:
        """Compute cache key from request content."""
        # Create a deterministic representation
        key_data = {
            "provider": provider,
            "model": model,
            "messages": request.messages,
            "tools": request.tools,
            "tool_choice": request.tool_choice,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return xxhash.xxh64(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path for a cache key, using subdirectories to avoid too many files."""
        return self.cache_dir / key[:2] / f"{key}.json"
    
    def get(self, provider: str, model: str, request: ModelRequest) -> ModelResponse | None:
        """Get cached response if available."""
        key = self._compute_key(provider, model, request)
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                self._hits += 1
                return ModelResponse(**data)
            except (json.JSONDecodeError, Exception):
                # Corrupted cache entry, remove it
                cache_path.unlink(missing_ok=True)
        
        self._misses += 1
        return None
    
    def set(
        self, 
        provider: str, 
        model: str, 
        request: ModelRequest, 
        response: ModelResponse,
    ) -> None:
        """Cache a response."""
        key = self._compute_key(provider, model, request)
        cache_path = self._get_cache_path(key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't cache responses with errors or tool errors
        if response.finish_reason == "error":
            return
        
        with open(cache_path, "w") as f:
            json.dump(response.model_dump(exclude={"raw_response"}), f)
    
    def invalidate(self, provider: str, model: str, request: ModelRequest) -> bool:
        """Remove a cached response."""
        key = self._compute_key(provider, model, request)
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False
    
    def clear(self) -> int:
        """Clear all cached responses. Returns count of removed entries."""
        count = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.json"):
                    cache_file.unlink()
                    count += 1
                # Remove empty subdirectory
                if not list(subdir.iterdir()):
                    subdir.rmdir()
        return count
    
    @property
    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
        }
    
    def size(self) -> int:
        """Count cached entries."""
        count = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                count += len(list(subdir.glob("*.json")))
        return count
