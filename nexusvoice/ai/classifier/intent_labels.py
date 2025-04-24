"""Constants for intent classification"""

from enum import Enum
from typing import List

class IntentLabels(Enum):
    """Constants for intent classification labels"""
    CONVERSATION = "conversation"
    HOME_AUTOMATION = "home_automation"
    
    @classmethod
    def all_labels(cls) -> List[str]:
        """Return list of all labels"""
        return [label.value for label in cls]
