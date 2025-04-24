import pytest
import re
from nexusvoice.ai.classifier.slots import (
    SlotValue, SlotType, Slot, SlotSet, SlotTypeSet, SampledSlotType, SlotSampler
)
from nexusvoice.ai.classifier.utterance import (
    StringFragment, StringFragmentSet, SlotFragment, OptionalFragment, Utterance
)

from nexusvoice.ai.classifier.dsl import S, U

# --- SlotValue ---
def test_slotvalue_basic():
    sv = SlotValue("on")
    assert sv.value == "on"
    assert sv.synonyms == set()
    sv.add_synonyms("activate", "turn on")
    assert "activate" in sv.synonyms
    assert "turn on" in sv.synonyms
    # Test __repr__
    assert "on" in repr(sv)
    assert "activate" in repr(sv)

def test_slotvalue_with_fragment():
    frag = StringFragment("enabled")
    sv = SlotValue("on")
    sv.add_synonyms(frag)
    assert "enabled" in sv.synonyms

# --- SlotType ---
def test_slottype_add_and_get():
    st = SlotType("device")
    st.add_values("fan", SlotValue("light", {"lamp"}))
    names = [v.value for v in st.get_values()]
    assert "fan" in names
    assert "light" in names
    # Test __str__ and __repr__
    assert "device" in str(st)
    assert "{device}" == str(st)
    assert "device" in repr(st)
    assert re.match(r"{device \[.*\]}", repr(st))

def test_slottype_synonyms():
    st = SlotType("action")
    sv = SlotValue("on").add_synonyms("activate", "switch on")
    st.add_values(sv)
    all_synonyms = set()
    for v in st.get_values():
        all_synonyms |= v.synonyms
    assert "activate" in all_synonyms
    assert "switch on" in all_synonyms

# --- SlotTypeSet ---
def test_slottype_set():
    st1 = SlotType("device").add_values("fan")
    st2 = SlotType("room").add_values("kitchen")
    sts = SlotTypeSet([st1, st2])
    values = [v.value for v in sts.get_values()]
    assert "fan" in values and "kitchen" in values
    assert "device" in sts.get_name()
    assert "room" in str(sts)

def test_sampled_slottype():
    st = SlotType("numbers").add_values(*[str(i) for i in range(100)])
    sampled = SampledSlotType(st, sample_count=10)
    vals = sampled.get_values()
    assert len(vals) == 10
    # All values must be from original
    all_vals = set(v.value for v in st.get_values())
    for v in vals:
        assert v.value in all_vals

def test_sampled_slottype_with_synonyms():
    st = SlotType('light_value')
    values = [
        ("1", ["1", "one", "number one", 'uno', 'één']),
        ("2", ["2", "two", "number two", 'dos', 'twee']),
        ("3", ["3", "three", "number three", 'tres', 'drie']),
        ("4", ["4", "four", "number four", 'cuatro', 'vier']),
        ("5", ["5", "five", "number five", 'cinco', 'vijf']),
        ("6", ["6", "six", "number six", 'seis', 'zes']),
        ("7", ["7", "seven", "number seven", 'siete', 'zeven']),
        ("8", ["8", "eight", "number eight", 'ocho', 'acht']),
        ("9", ["9", "nine", "number nine", 'nueve', 'negen']),
        ("10", ["10", "ten", "number ten", 'diez', 'tien']),
    ]
    for value, synonyms in values:
        st.add_values(SlotValue(value, set(synonyms)))
    sampled = SampledSlotType(st, sample_count=2)
    vals = sampled.get_values()
    assert len(vals) == 2
    # All values must be from original
    all_vals = set(v.value for v in st.get_values())
    for v in vals:
        assert v.value in all_vals

    # Check permuted with synonyms
    samples = SampledSlotType(st, sample_count=2)
    vals = samples.get_permutations()
    assert len(vals) == 2
    # All values must be from original
    all_vals = set(v for v in st.get_permutations())
    for v in vals:
        assert v in all_vals

# --- Slot & SlotSet ---
def test_slot_and_slotset():
    st = SlotType("device").add_values("fan", "light")
    slot = Slot("target", st)
    assert slot.get_name() == "target"
    assert slot.get_type() is st
    ss = SlotSet([slot])
    assert slot in ss.get_slots()
    assert "target" in ss.get_name()
    assert "fan" in [v.value for v in ss.get_values()]
    # Add more slots
    slot2 = Slot("room", SlotType("room").add_values("kitchen"))
    ss.add(slot2)
    assert slot2 in ss.get_slots()
    assert "room" in ss.get_name()

# --- SlotSampler ---
def test_slotsampler():
    st = SlotType("numbers").add_values(*[str(i) for i in range(50)])
    slot = Slot("num", st)
    sampled_slot = SlotSampler.sample(slot, sample_count=5)
    vals = sampled_slot.get_type().get_values()
    assert len(vals) == 5
    all_vals = set(v.value for v in st.get_values())
    for v in vals:
        assert v.value in all_vals


def test_slottype_with_utterance_synonyms():
    st = SlotType('light_value')
    values = [
        ("1", ['one',   U(S("number one", 'uno', 'één'))]),
        ("2", ['two',   U(S("number two", 'dos', 'twee'))]),
        ("3", ['three', U(S("number three", 'tres', 'drie'))]),
        ("4", ['four',  U(S("number four", 'cuatro', 'vier'))]),
        ("5", ['five',  U(S("number five", 'cinco', 'vijf'))]),
        ("6", ['six',   U(S("number six", 'seis', 'zes'))]),
        ("7", ['seven', U(S("number seven", 'siete', 'zeven'))]),
        ("8", ['eight', U(S("number eight", 'ocho', 'acht'))]),
        ("9", ['nine',  U(S("number nine", 'nueve', 'negen'))]),
        ("10", ['ten',  U(S("number ten", 'diez', 'tien'))]),
    ]
    for value, synonyms in values:
        for synonym in synonyms:
            sv = SlotValue(value)
            sv.add_synonyms(synonym)
            st.add_values(sv)
    sampled = SampledSlotType(st, sample_count=2)
    vals = sampled.get_values()
    assert len(vals) == 2
    # All values must be from original
    all_vals = set(v.value for v in st.get_values())
    for v in vals:
        assert v.value in all_vals

    # Check permuted with synonyms
    samples = SampledSlotType(st, sample_count=2)
    vals = samples.get_permutations()
    assert len(vals) == 2
    # All values must be from original
    all_vals = set(v for v in st.get_permutations())
    for v in vals:
        assert v in all_vals
    

# --- StringFragment & StringFragmentSet ---
def test_stringfragment():
    frag = StringFragment("hello")
    assert frag.text == "hello"
    assert frag.get_permutations() == [frag]
    assert str(frag) == "hello"

def test_stringfragmentset():
    sfs = StringFragmentSet("hi", "hello")
    perms = sfs.get_permutations()
    texts = set(str(f) for f in perms)
    assert "hi" in texts and "hello" in texts
    assert "hi" in str(sfs) and "hello" in str(sfs)

# --- SlotFragment ---
def test_slotfragment():
    st = SlotType("device").add_values("fan", SlotValue("light", {"lamp"}))
    slot = Slot("target", st)
    frag = SlotFragment(slot)
    perms = frag.get_permutations()
    texts = set(str(f) for f in perms)
    assert "fan" in texts and "light" in texts and "lamp" in texts
    assert "target" in str(frag)

# --- OptionalFragment ---
def test_optionalfragment():
    frag = StringFragment("please")
    opt = OptionalFragment(frag)
    perms = opt.get_permutations()
    assert perms[0] is None
    assert isinstance(perms[1], StringFragment)
    assert str(opt) == "[please]"

# --- Utterance ---
def test_utterance_basic():
    st = SlotType("device").add_values("fan", "light")
    slot = Slot("target", st)
    utt = Utterance("turn", slot, "on")
    assert len(utt.fragments) == 3
    assert isinstance(utt.fragments[1], SlotFragment)
    assert "turn" in str(utt)
    assert "on" in str(utt)

def test_utterance_permutations():
    st = SlotType("device").add_values("fan", SlotValue("light", {"lamp"}))
    slot = Slot("target", st)
    utt = Utterance("turn", slot, "on")
    perms = utt.get_permutations()
    # Should generate all combinations: 2 devices x 1 x 1 = 2, plus synonyms
    texts = set(str(u) for u in perms)
    assert any("fan" in t for t in texts)
    assert any("light" in t for t in texts)
    assert any("lamp" in t for t in texts)
    assert all("turn" in t and "on" in t for t in texts)

def test_utterance_with_optional():
    st = SlotType("device").add_values("fan")
    slot = Slot("target", st)
    opt = OptionalFragment(StringFragment("please"))
    utt = Utterance(opt, "turn", slot, "on")
    perms = utt.get_permutations()
    # Should include versions with and without "please"
    texts = set(str(u) for u in perms)
    assert any(t.startswith("please") for t in texts)
    assert any(not t.startswith("please") for t in texts)
