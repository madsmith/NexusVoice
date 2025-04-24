from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import List, Set, Optional, Union

from nexusvoice.ai.classifier.base_types import UtteranceFragment

@dataclass
class SlotValue:
    """A value that can be used in a slot"""
    value: str
    synonyms: Set[str] = field(default_factory=set)
    
    def add_synonyms(self, *synonyms: Union[str, UtteranceFragment]) -> 'SlotValue':
        """Add synonyms for this value"""
        for syn in synonyms:
            if isinstance(syn, UtteranceFragment):
                for frag in syn.permutations():
                    if frag is not None:
                        self.synonyms.add(str(frag))
            else:
                self.synonyms.add(syn)
        return self

    def __str__(self) -> str:
        if self.synonyms:
            return f"~\"{self.value}\""
        return f"\"{self.value}\""

    def __repr__(self) -> str:
        if self.synonyms:
            return f"\"{self.value}\" ({', '.join(self.synonyms)})"
        return f"\"{self.value}\""

class SlotTypeLike(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this slot"""
        raise NotImplementedError
    
    def get_values(self) -> List[SlotValue]:
        """Get all values for this slot type"""
        raise NotImplementedError

    def get_permutations(self) -> List[str]:
        """Generate permutations using all values from the slot type"""
        values = []
        
        for slot_value in self.get_values():
            # Add the main value
            values.append(slot_value.value)
            # Add all synonyms
            values.extend(slot_value.synonyms)

        return values

@dataclass
class SlotLike(ABC):
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this slot"""
        pass
    
    @abstractmethod
    def get_type(self) -> SlotTypeLike:
        """Get the type of this slot"""
        pass
    
    def get_values(self) -> List[SlotValue]:
        """Get all values for this slot type"""
        return self.get_type().get_values()

    def get_permutations(self) -> List[str]:
        """Generate permutations using all values from the slot type"""
        return self.get_type().get_permutations()

class SlotType(SlotTypeLike):
    """The type of data that a slot can hold."""

    def __init__(self, name: str, values: Optional[List[SlotValue]] = None):
        self.name: str = name
        self.values: List[SlotValue] = values or []
    
    def get_name(self) -> str:
        """Get the name of this slot type"""
        return self.name

    def get_values(self) -> List[SlotValue]:
        """Get all values for this slot type"""
        return self.values

    def add_values(self, *values: Union[str, SlotValue]) -> 'SlotType':
        """Add values to this slot type"""
        for value in values:
            if isinstance(value, str):
                self.values.append(SlotValue(value))
            elif isinstance(value, SlotValue):
                self.values.append(value)
        return self

    def __str__(self) -> str:
        return f"{{{self.name}}}"
    
    def __repr__(self) -> str:
        return f"{{{self.name} [{', '.join(str(v) for v in self.values)}]}}"

class SlotTypeSet(SlotTypeLike):
    """A set of slot types"""
    def __init__(self, slot_types: List[SlotTypeLike]):
        self.slot_types = slot_types

    def get_name(self) -> str:
        """Get the name of this slot type set"""
        return f"Set({'|'.join(s.get_name() for s in self.slot_types)})"
    
    def get_values(self) -> List[SlotValue]:
        """Get all values for this slot type set"""
        # Contiguous values flattened
        values = []
        for slot_type in self.slot_types:
            values.extend(slot_type.get_values())
        return values

    def add(self, *slot_types: SlotTypeLike) -> 'SlotTypeSet':
        """Add slot types to this set"""
        self.slot_types.extend(slot_types)
        return self

    def __str__(self) -> str:
        return f"{{{'|'.join(s.get_name() for s in self.slot_types)}}}"

    def __repr__(self) -> str:
        return f"{{{'|'.join(s.get_name() for s in self.slot_types)}}}"

class SampledSlotType(SlotTypeLike):
    def __init__(self, slot_type: SlotTypeLike, sample_count: int = 20):
        self.slot_type = slot_type
        self.sample_count = sample_count

    def get_name(self) -> str:
        return self.slot_type.get_name()

    def get_raw_slot_type(self) -> SlotTypeLike:
        return self.slot_type
    
    def _sample(self, data):
        return random.sample(data, min(self.sample_count, len(data)))

    def get_values(self) -> List[SlotValue]:
        values = self.slot_type.get_values()
        return self._sample(values)

    def get_permutations(self) -> List[str]:
        values = []
        # Use the underlying slot for values as get_values is sampled
        for slot_value in self.slot_type.get_values():
            values.append(slot_value.value)
            values.extend(slot_value.synonyms)

        return self._sample(values)

    def __str__(self) -> str:
        values = self.slot_type.get_values()
        if len(values) < 5:
            return f"{{{'|'.join(s.value for s in values)}}}"
        else:
            return f"{{{'|'.join(s.value for s in values[:5])}...}}"

    def __repr__(self) -> str:
        values = self.slot_type.get_values()
        if len(values) < 5:
            return f"{{{'|'.join(repr(s) for s in values)}}}"
        else:
            return f"{{{'|'.join(repr(s) for s in values[:5])}...}}"

@dataclass
class Slot(SlotLike):
    """A slot that can be filled with a value"""
    name: str
    slot_type: SlotTypeLike
    
    def get_name(self) -> str:
        """Get the name of this slot"""
        return self.name
    
    def get_type(self) -> SlotTypeLike:
        """Get the type of this slot"""
        return self.slot_type


@dataclass
class SlotSet(SlotLike):
    """A set of slots"""
    slots: List[Slot] = field(default_factory=list)
    
    def get_name(self) -> str:
        """Get the name of this slot set"""
        return f"Set({', '.join(s.get_name() for s in self.slots)})"
    
    def get_type(self) -> SlotTypeLike:
        """Get the type of this slot set"""
        return SlotTypeSet([slot.get_type() for slot in self.slots])
    
    def add(self, *slots: Slot) -> 'SlotSet':
        """Add slots to this set"""
        for slot in slots:
            self.slots.append(slot)
        return self
    
    def get_slots(self) -> List[Slot]:
        """Get all slots in this set"""
        return self.slots
    
    def get_values(self) -> List[SlotValue]:
        """Get all values for this slot set"""
        # Contiguous values flattened
        values = []
        for slot in self.slots:
            values.extend(slot.get_values())
        return values

class SlotSampler():
    @staticmethod
    def sample(slot: Slot, sample_count: int = 20) -> Slot:
        """ Return a new slot who's type is a sampled version of the original slot type """
        slot_type = SampledSlotType(slot.get_type(), sample_count)
        return Slot(slot.get_name(), slot_type)