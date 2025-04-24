from dataclasses import dataclass, field
import random
from typing import List, Set, Union, Optional
import itertools

from .slots import SampledSlotType, Slot, SlotLike, SlotSet
from nexusvoice.ai.classifier.base_types import UtteranceFragment

@dataclass
class StringFragment(UtteranceFragment):
    """A simple string fragment"""
    text: str

    def __init__(self, text: str):
        super().__init__()
        self.text = text
    
    def get_permutations(self) -> List[Optional[UtteranceFragment]]:
        return [self]
    
    def __str__(self) -> str:
        return self.text

@dataclass
class StringFragmentSet(UtteranceFragment):
    """A set of alternative string fragments"""
    fragments: Set[str] = field(default_factory=set)

    def __init__(self, *args):
        super().__init__()
        self.fragments = set()
        self.add(*args)
    
    def add(self, *values: Union[str, List[str]]) -> 'StringFragmentSet':
        for value in values:
            if isinstance(value, list):
                self.add(*value)
            else:
                self.fragments.add(value)
        return self
    
    def get_permutations(self) -> List[Optional[UtteranceFragment]]:
        return [StringFragment(s) for s in self.fragments]
    
    def __str__(self) -> str:
        return f"({'|'.join(self.fragments)})"

@dataclass
class SlotFragment(UtteranceFragment):
    """A fragment representing a slot"""
    slot: SlotLike

    def __init__(self, slot: SlotLike):
        super().__init__()
        self.slot = slot
    
    def get_permutations(self) -> List[Optional[UtteranceFragment]]:
        """Generate permutations using all values from the slot type"""
        values = self.slot.get_permutations()
        fragments: List[Optional[UtteranceFragment]] = [StringFragment(v) for v in values]

        return fragments
    
    def __str__(self) -> str:
        if isinstance(self.slot, SlotSet):
            return f"S({'|'.join(f"{s.get_name()}" for s in self.slot.get_slots())})"

        return f"{{{self.slot.get_name()}}}"


@dataclass
class OptionalFragment(UtteranceFragment):
    """A fragment that can be omitted"""
    fragment: UtteranceFragment

    def __init__(self, fragment: UtteranceFragment):
        super().__init__()
        self.fragment = fragment
    
    def get_permutations(self) -> List[Optional[UtteranceFragment]]:
        # Return both with and without the fragment
        return [None] + self.fragment.permutations()
    
    def __str__(self) -> str:
        return f"[{self.fragment}]"

@dataclass
class Utterance(UtteranceFragment):
    """A complete utterance made up of fragments"""
    fragments: List[UtteranceFragment] = field(default_factory=list)
    
    def __init__(self, *args):
        super().__init__()
        self.fragments = []
        self.add(*args)
    
    def add(self, *args: Union[str, Slot, SlotSet, UtteranceFragment]) -> 'Utterance':
        for arg in args:
            if isinstance(arg, str):
                self.fragments.append(StringFragment(arg))
            elif isinstance(arg, UtteranceFragment):
                self.fragments.append(arg)
            elif isinstance(arg, Slot):
                self.fragments.append(SlotFragment(arg))
            elif isinstance(arg, SlotSet):
                self.fragments.append(SlotFragment(arg))
            elif arg is None:
                continue
            else:
                raise TypeError(f"Cannot add {type(arg)} to utterance")
        return self
    
    def get_permutations(self) -> List[Optional[UtteranceFragment]]:
        # Get permutations for each fragment
        fragment_perms = [f.permutations() if f else [None] for f in self.fragments]
        
        # Get cartesian product of all permutations
        all_perms = list(itertools.product(*fragment_perms))
        
        # Create new utterances from permutations
        result = []
        for perm in all_perms:
            # Filter out None values (from OptionalFragments)
            filtered = [p for p in perm if p is not None]
            if filtered:  # Only add if there are fragments left
                u = Utterance()
                u.fragments = filtered
                result.append(u)
        
        return result
    
    def __str__(self) -> str:
        return ' '.join(str(f) for f in self.fragments if f is not None)
