import random
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from nexusvoice.ai.classifier.base_types import UtteranceFragment
from nexusvoice.ai.classifier.dsl import U, S, O, sv
from nexusvoice.ai.classifier.slots import SlotSampler, SlotSet, SlotType, Slot
from nexusvoice.ai.classifier.utterance import SlotFragment


def slot_type_light_direction() -> SlotType:
    # Light direction
    light_direction = SlotType('LIGHT_DIRECTION')
    light_direction.add_values(
        sv('on'),
        sv('off')
    )
    return light_direction

def slot_type_shade_direction() -> SlotType:
    # Shade direction
    shade_direction = SlotType('SHADE_DIRECTION')
    shade_direction.add_values(
        sv('open').add_synonyms('up', 'raise'),
        sv('close').add_synonyms('down', 'lower')
    )
    return shade_direction

def slot_type_cardinal_direction() -> SlotType:
    # Cardinal directions
    cardinal_direction = SlotType('CARDINAL_DIRECTION')
    cardinal_direction.add_values(
        sv('north'), sv('south'), sv('east'), sv('west')
    )
    return cardinal_direction

def slot_type_level() -> SlotType:
    # Light level (numerical)
    level = SlotType('LEVEL')
    level.add_values(*[sv(str(i)) for i in range(100)])
    return level

def slot_type_room() -> SlotType:
    # Rooms
    room_name = SlotType('ROOM_NAME')
    
    # Basic rooms
    # These rooms follow the format: {room name} room
    basic_rooms = [
        'project', 'laundry', 'breakfast', 'powder', 'dining', 'family', 'living',
        'secret', 'server', 'exercise'
    ]
    for room in basic_rooms:
        room_name.add_values(sv(room).add_synonyms(f'{room} room'))
    
    # Special rooms
    # These rooms typically do not have a room suffix when being referenced
    room_name.add_values(
        sv('mudroom').add_synonyms('mud', 'mud room'),
        sv('playroom').add_synonyms('play', 'play room'),
        sv('master').add_synonyms('master bed', 'master bedroom'),
        sv('master closet').add_synonyms(
            'master bedroom closet', 'master walk-in', 'master walk in',
            'master bedroom walk-in', 'master bedroom walk in'
        ),
        sv('master bath').add_synonyms('master bathroom'),
        sv('guest 1').add_synonyms('guest bedroom 1'),
        sv('guest 2').add_synonyms('guest bedroom 2'),
        sv('office'),
        sv('entry').add_synonyms('foyer'),
        sv('stairwell').add_synonyms('stair', 'stairs', 'stairway'),
        sv('screened porch').add_synonyms('screen porch'),
        sv('sidewalk').add_synonyms('front sidewalk')
    )
    
    # Other rooms
    # Simple extensions to special rooms
    other_rooms = [
        'garage', 'gallery', 'kitchen', 'lounge', 'play alcove',
        'pool bath', 'theater', 'perimeter', 'landscape',
        'hallway', 'master balcony', 'balcony'
    ]
    for room in other_rooms:
        room_name.add_values(sv(room))
    
    return room_name

def slot_type_light() -> SlotType:
    # Light name
    light_name = SlotType('LIGHT_NAME')

    recessedCans = U(O('recessed'), O(S('can', 'cans')))
    sconces = U(S('sconce', 'sconces'))
    fans = U(S('fan', 'fans'))

    light_name.add_values(
        sv('recessed').add_synonyms(recessedCans),
        sv('hallway').add_synonyms('hall', 'walkway', recessedCans),
        sv('patio').add_synonyms(U(O('under'), 'patio', O(recessedCans)))
    )

    light_locators = [
        'center', 'perimeter', 'wall wash', 'fireplace', 'landing',
        'shower', 'tub', 'front', 'rear', 'island', 'vestibule',
        'art niche', 'balcony', 'eave', 'sink', 'soffit'
    ]
    for locator in light_locators:
        light_name.add_values(sv(locator).add_synonyms(U(locator, recessedCans)))

    light_name.add_values(
        sv('shelf').add_synonyms('under shelf'),
        sv('cabinet').add_synonyms('under cabinet'),
        sv('table perimeter').add_synonyms('table perimeter', recessedCans),
        sv('table center').add_synonyms('table center', recessedCans),
        sv('table chandelier')
    )

    light_name.add_values(sv('pendant').add_synonyms(U(S('center', 'island', 'table'), 'pendant')))

    light_name.add_values(sv('work bench l. e. d. strip').add_synonyms(U('work bench', O('l. e. d. '), O('strip'))))
    light_name.add_values(sv('slop sink l. e. d. strip').add_synonyms(U(O('slop'), 'sink', O('l. e. d. '), O('strip'))))
    light_name.add_values(sv('counter l. e. d. strip').add_synonyms(U('counter', O('l. e. d. '), O('strip'))))

    # Toe Kick
    light_name.add_values(sv('toe kick').add_synonyms('vanity toe kick'))

    # Sconces
    light_name.add_values(sv('vanity sconce').add_synonyms(U('vanity', O('wall'), sconces)))
    light_name.add_values(sv('sconce').add_synonyms(U(O('wall'), sconces)))

    # Fans
    light_name.add_values(sv('ceiling fan').add_synonyms('ceiling fans'))
    light_name.add_values(sv('exhaust fan').add_synonyms(U(S('toilet', 'shower'), O('exhaust'), fans)))
    light_name.add_values(sv('patio fan').add_synonyms(U(O('under'), 'patio', fans)))

    # Lamps
    light_name.add_values(sv('his lamp').add_synonyms(U('his', S('bedside', 'bed', 'table', 'nightstand', 'reading'), 'lamp')))
    light_name.add_values(sv('her lamp').add_synonyms(U('her', S('bedside', 'bed', 'table', 'nightstand', 'reading'), 'lamp')))

    # Coach Lights
    light_name.add_values(sv('exercise patio coach').add_synonyms('exercise coach', 'patio coach'))
    light_name.add_values(sv('balcony coach'))
    light_name.add_values(sv('garage coach'))

    # Other
    light_name.add_values(sv('bedroom').add_synonyms('room'))
    light_name.add_values('linear', 'pathway', 'step')
    light_name.add_values(sv('bridge soffit').add_synonyms('bridge', 'soffit'))
    light_name.add_values(sv('closet').add_synonyms(U(S('left', 'right'), 'closet')))
    light_name.add_values(sv('mirror backlights').add_synonyms('mirror back lights'))
    light_name.add_values('mirror defoggers')
    light_name.add_values(*[f'landscape {n}' for n in ['one', 'two', 'three', 'four', '1', '2', '3', '4']])
    light_name.add_values(sv('bollard').add_synonyms('post bollard'))

    return light_name

def slot_type_shade() -> SlotType:
    # Shade name
    shade_name = SlotType('SHADE_NAME')
    shade_name.add_values('blackout', 'sheer', 'door', 'window')
    for direction in ['north', 'south', 'east', 'west']:
        shade_name.add_values(sv(f"{direction} blackout"))
        shade_name.add_values(sv(f"{direction} sheer"))
        for suffix in ["one", "1", "two", "2", "three", "3", "four", "4"]:
            shade_name.add_values(sv(f"{direction} blackout {suffix}"))
            shade_name.add_values(sv(f"{direction} sheer {suffix}"))

    return shade_name

def slot_type_floor() -> SlotType:
    # Floor name
    floor_name = SlotType('FLOOR_NAME')
    floor_name.add_values(
        sv('first').add_synonyms('main'),
        sv('second').add_synonyms('upstairs'),
        sv('lower level').add_synonyms('basement', 'downstairs'),
        sv('exterior').add_synonyms('outside')
    )
    return floor_name

def slot_type_person() -> SlotType:
    # Person name
    person_name = SlotType('PERSON_NAME')
    return person_name

# Define slot types
def create_slot_types() -> Dict[str, SlotType]:
    # Return the slot types
    return {
        'light_direction': slot_type_light_direction(),
        'shade_direction': slot_type_shade_direction(),
        'cardinal_direction': slot_type_cardinal_direction(),
        'light_level': slot_type_level(),
        'room_name': slot_type_room(),
        'light_name': slot_type_light(),
        'shade_name': slot_type_shade(),
        'floor_name': slot_type_floor(),
        'person_name': slot_type_person()
    }

    
def generate_home_automation_examples() -> List[str]:
    '''Generate home automation examples using the DSL'''
    slot_types = create_slot_types()
    
    # Create slots
    # Location Slots
    floor_name = Slot('FloorName', slot_types['floor_name'])
    room_name = Slot('RoomName', slot_types['room_name'])
    # Fixture Slots
    light_name = SlotSampler.sample(Slot('LightName', slot_types['light_name']), sample_count=50)
    shade_name = Slot('ShadeName', slot_types['shade_name'])
    # Value Slots
    light_direction = Slot('LightDirection', slot_types['light_direction'])
    level = Slot('LightLevel', slot_types['light_level'])
    shade_direction = Slot('ShadeDirection', slot_types['shade_direction'])
    cardinal_direction = Slot('Direction', slot_types['cardinal_direction'])
    person_name = Slot('PersonName', slot_types['person_name'])

    light_value = SlotSet([light_direction, SlotSampler.sample(level)])

    
    # Common fragments
    lights = S('light', 'lights')
    in_on = S('in', 'on')
    toAt = S('to', 'at')
    turn_set = S('turn', 'set', 'make')

    # Generate utterances for light control
    utterances = []
    
    # Basic light control patterns
    # TODO: Pattern for person_name room_name lights
    patterns = [
        U('set', O('the'), O(floor_name), room_name, O(light_name), O(lights), toAt, light_value),
        U('turn', O('the'), O(floor_name), room_name, O(light_name), O(lights), 'to', light_value),
        U('turn', light_direction, O('the'), floor_name, light_name, O(lights)),
        U('turn', light_direction, O('the'), O(floor_name), room_name, O(light_name), O(lights)),
        U('turn', light_direction, O('the'), lights, in_on, 'the', O(floor_name), room_name),
        U(O(floor_name), room_name, light_name, light_direction),
        U(turn_set, 'all', O('the'), lights, in_on, 'the', floor_name, O('floor'), toAt, light_value),
        U(turn_set, 'all', O('the'), lights, in_on, 'the', floor_name, O('floor'), light_direction),
        U('turn', light_direction, 'all', O('the'), lights, in_on, 'the', floor_name, O('floor')), 
        U('turn', light_direction, 'all the', floor_name, O('floor'), lights)
    
        # # Turn on the lights in the [first floor] kitchen
        # U(turn_set, light_direction, the, lights, in_on, O(the), O(floor_name),room_name),
        # # Turn on the [first floor] kitchen [type] lights
        # U(turn_set, light_direction, the, O(floor_name), room_name, O(light_name), O(lights)),
        # # Set the [first floor] kitchen [type] lights to value
        # U(set, the, O(floor_name), room_name, O(light_name), O(lights), toAt, light_value),
        # # U(turn_set, light_direction, the, room_name, O(lights)),
        # U(turn_set, the, room_name, lights, light_direction),
        # U(light_direction, the, lights, in_on, the, room_name),
        # U(light_direction, the, room_name, lights),
    ]
    
    # Generate all permutations
    for pattern in patterns:
        start_count = len(utterances)
        print("== Generating permutations for pattern:\n  ", pattern)
        utterances.extend([str(u) for u in pattern.permutations()])
        print(f"==Generated {len(utterances) - start_count} utterances [Total: {len(utterances)}]")
    
    return utterances

def generate_conversation_examples() -> List[str]:
    '''Generate general conversation examples'''
    # Weather patterns
    weather = [
        U("what's", 'the', 'weather', O('like'), O('today')),
        U("how's", 'the', 'weather', O('today')),
        U('is', 'it', 'going', 'to', 'rain', O('today')),
        U("what's", 'the', 'forecast', O('for'), O('today')),
    ]
    
    # Time patterns
    time = [
        U('what', 'time', 'is', 'it'),
        U("what's", 'the', 'time'),
        U('tell', 'me', 'the', 'time'),
    ]
    
    # General questions
    questions = [
        U('how', 'are', 'you', O('today')),
        U('what', 'can', 'you', 'do'),
        U('tell', 'me', 'a', 'joke'),
        U('who', 'are', 'you'),
        U("what's", 'your', 'name'),
    ]
    
    # Combine all patterns
    all_patterns = weather + time + questions
    
    # Generate all permutations
    utterances = []
    for pattern in all_patterns:
        utterances.extend([str(u) for u in pattern.permutations()])
    
    return utterances

def generate_training_data() -> Tuple[List[str], List[int]]:
    '''Generate training data with labels'''
    home_automation = generate_home_automation_examples()
    conversation = generate_conversation_examples()
    
    # Create labels (1 for home automation, 0 for conversation)
    labels = [1] * len(home_automation) + [0] * len(conversation)
    
    # Combine examples
    examples = home_automation + conversation
    
    return examples, labels

if __name__ == '__main__':

    ROOT_DIR = Path(__file__).parent.parent.parent.parent
    BUILD_DIR = ROOT_DIR / 'build' / 'data'
    
    # Generate examples
    examples, labels = generate_training_data()
    
    # Print some statistics
    print(f'Generated {len(examples)} examples:')
    print(f'- Home automation: {sum(labels)}')
    print(f'- Conversation: {len(labels) - sum(labels)}')
    
    # Save examples and labels
    BUILD_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Saving examples and labels to {BUILD_DIR}")
    with open(BUILD_DIR / 'home_automation_examples.txt', 'w') as f:
        f.write('\n'.join(examples))
    with open(BUILD_DIR / 'home_automation_labels.txt', 'w') as f:
        f.write('\n'.join(map(str, labels)))
    
    # Print some examples
    print('\nSample home automation examples:')
    ha_examples = [e for e, l in zip(examples, labels) if l == 1]
    random.shuffle(ha_examples)
    for example in ha_examples[:5]:
        print(f'- {example}')
    
    print('\nSample conversation examples:')
    conv_examples = [e for e, l in zip(examples, labels) if l == 0]
    random.shuffle(conv_examples)
    for example in conv_examples[:5]:
        print(f'- {example}')
