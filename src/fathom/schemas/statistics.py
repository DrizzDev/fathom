from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class CacheStats(BaseModel):
    """
    Tracks cache performance metrics.
    """

    model_config = ConfigDict(frozen=False)

    hits: int = Field(default=0, description="Number of cache hits")
    misses: int = Field(default=0, description="Number of cache misses")
    evictions: int = Field(default=0, description="Number of caches evicted")
    creates: int = Field(default=0, description="Number of new caches created")

    @property
    def hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        """

        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert stats to dictionary format.
        """

        return {
            "hits": self.hits,
            "misses": self.misses,
            "creates": self.creates,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 3),
        }
