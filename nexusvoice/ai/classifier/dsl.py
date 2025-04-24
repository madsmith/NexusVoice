from typing import Union

from .slots import Slot, SlotValue
from .utterance import SlotFragment, Utterance, StringFragmentSet, OptionalFragment, StringFragment, UtteranceFragment

def sv(name: str) -> SlotValue:
    """Create a new slot value (DSL shorthand)"""
    return SlotValue(name)

def U(*args) -> Utterance:
    """Create a new utterance"""
    return Utterance(*args)

def S(*args) -> StringFragmentSet:
    """Create a set of alternative strings"""
    return StringFragmentSet().add(*args)

def O(fragment: Union[str, Slot, UtteranceFragment]) -> OptionalFragment:
    """Create an optional fragment"""
    if isinstance(fragment, str):
        fragment = StringFragment(fragment)
    elif isinstance(fragment, Slot):
        fragment = SlotFragment(fragment)
    return OptionalFragment(fragment)
