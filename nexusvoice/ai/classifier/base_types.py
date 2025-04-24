from abc import ABC, abstractmethod
from typing import List, Optional

class UtteranceFragment(ABC):
    """Base class for all utterance fragments with cached permutations"""
    def __init__(self):
        self._permutations_cache = None

    def permutations(self) -> List[Optional['UtteranceFragment']]:
        if self._permutations_cache is None:
            self._permutations_cache = self.get_permutations()
        return self._permutations_cache

    @abstractmethod
    def get_permutations(self) -> List[Optional['UtteranceFragment']]:
        """Return all possible permutations of this fragment (to be implemented by subclasses)"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the fragment"""
        pass
