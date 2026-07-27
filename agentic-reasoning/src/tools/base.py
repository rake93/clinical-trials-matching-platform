from abc import ABC, abstractmethod
import json
from typing import Any

from cachetools import TTLCache


class BaseTool(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        ttl = config.get("cache_ttl", 300)
        maxsize = config.get("cache_maxsize", 128)
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)

    @abstractmethod
    def execute(self, input: Any) -> Any:
        """Execute the tool with given input."""
        ...

    def cached_execute(self, input: Any) -> Any:
        """Execute with TTL caching keyed on a canonical representation of input."""
        key = (
            input
            if isinstance(input, str)
            else json.dumps(input, sort_keys=True, separators=(",", ":"))
        )
        if key not in self._cache:
            self._cache[key] = self.execute(input)
        return self._cache[key]

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...
