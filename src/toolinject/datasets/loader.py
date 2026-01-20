"""Base dataset loader."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from toolinject.core.schemas import TestCase, AttackCategory


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    name: str = "base"
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    @abstractmethod
    def load(self) -> list[TestCase]:
        """Load all test cases."""
        ...
    
    def load_subset(
        self,
        max_samples: int | None = None,
        categories: list[AttackCategory] | None = None,
        seed: int = 42,
    ) -> list[TestCase]:
        """Load a subset of test cases."""
        import random
        
        cases = self.load()
        
        # Filter by category
        if categories:
            cases = [c for c in cases if c.category in categories]
        
        # Sample if needed
        if max_samples and len(cases) > max_samples:
            random.seed(seed)
            cases = random.sample(cases, max_samples)
        
        return cases
    
    def iterate(self) -> Iterator[TestCase]:
        """Iterate over test cases."""
        yield from self.load()
    
    def count(self) -> int:
        """Count test cases."""
        return len(self.load())
    
    def categories(self) -> set[AttackCategory]:
        """Get unique categories in dataset."""
        return {c.category for c in self.load()}
