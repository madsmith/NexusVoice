from dataclasses import dataclass, field
import logging
import re

from nexusvoice.core.config import NexusConfig

from .database import LutronDatabase

@dataclass
class LutronAreaGroup:
    name: str
    
@dataclass
class LutronArea:
    iid: int
    name: str

    @classmethod
    def from_dict(cls, data):
        return cls(data['iid'], data['name'])
    
@dataclass
class LutronOutput:
    iid: int
    output_type: str
    name: str
    path: str

@dataclass
class LutronDBEntity:
    id: int
    iid: int
    name: str
    spoken_name: str | None
    type: str
    subtype: str | None
    path: str
    parent_id: int | None

@dataclass
class LutronEntity:
    iid: int
    name: str
    spoken_name: str | None
    type: str
    subtype: str | None
    path: str

    @classmethod
    def from_dict(cls, data):
        subtype = data['subtype'] if 'subtype' in data else None
        spoken_name = data['spoken_name'] if 'spoken_name' in data else None
        return cls(data['iid'], data['name'], spoken_name, data['type'], subtype, data['path'])
    
    @classmethod
    def from_db_entity(cls, entity: LutronDBEntity):
        return cls(entity.iid, entity.name, entity.spoken_name, entity.type, entity.subtype, entity.path)
    
    def __str__(self):
        return f"[{self.type}] {"(" +str(self.iid) + ")":<6} {self.name:<38} => {self.spoken_name:<20}"

@dataclass
class ObjectHolder:
    value: dict = field(default_factory=dict)
    id: int = field(init=False)

    _counter: int = 0  # class-level counter

    def __post_init__(self):
        type(self)._counter += 1
        self.id = type(self)._counter

@dataclass
class IIDMapRecord:
    id: int
    object: ObjectHolder
    parent: int | None

class LutronDatabaseView:
    _available_filters = {}
    
    @classmethod
    def register_filter(cls, filter_class: type):
        cls._available_filters[filter_class.filter_name] = filter_class

    def __init__(self, config: NexusConfig, database: LutronDatabase):
        self.config = config
        self.database = database
        self._filters = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._entities: dict[int, LutronDBEntity] = {}
        self._id_map: dict[int, IIDMapRecord] = {}

    def initialize(self):
        # Initialize filters based on config and available filters
        for filter_type, filter_rules in self.config.get("tools.lutron.filters").items():
            filter_class = self._available_filters.get(filter_type)
            if filter_class is None:
                self.logger.warning(f"Unknown filter type: {filter_type}")
                continue
            
            for filter_rule in filter_rules:
                self._filters.append(filter_class(*filter_rule))

        self._process_database()

        return self

    def _process_database(self):
        # Load data from the database
        db_iid_map = self.database.getIIDMap()
        # Play objects in ObjectHolder so they all have a distinct ID
        areas = [ObjectHolder(area) for area in self.database.getAreas()]
        outputs = [ObjectHolder(output) for output in self.database.getOutputs()]
        
        # Rebuild iid_map on DataHolder using each objects DataHolder.id
        self._rebuild_iid_map(db_iid_map, areas, outputs)

        # Rebuild entities
        self._rebuild_entities(areas, outputs)
        
    def _rebuild_iid_map(self, db_iid_map, areas: list[ObjectHolder], outputs: list[ObjectHolder]):
        def find_object(iid, name):
            obj = None
            for area in areas:
                if area.value['iid'] == iid and area.value['name'] == name:
                    obj = area
                    break
            if obj is None:
                for output in outputs:
                    if output.value['iid'] == iid and output.value['name'] == name:
                        obj = output
                        break
            return obj

        iid_map = {}
        for record in db_iid_map.values():
            obj_type = record['type']
            if obj_type not in self.config.get("tools.lutron.valid_object_types"):
                continue
            obj = find_object(record['iid'], record['name'])
            if obj is None:
                raise ValueError(f"No object found for: {record['iid']} {record['name']} {record['type']}")
            parent = None
            if record['parent_iid'] is not None or record['parent_name'] is not None:
                parent = find_object(record['parent_iid'], record['parent_name'])
                if parent is None:
                    raise ValueError(f"No parent found for: {record['parent_iid']} {record['parent_name']} {record['parent_type']}")
            else:
                pass

            parent_id = parent.id if parent is not None else None
            iid_map[obj.id] = IIDMapRecord(obj.id, obj, parent_id)
        
        self._id_map = iid_map
        
    def _rebuild_entities(self, areas: list[ObjectHolder], outputs: list[ObjectHolder]) -> None:
        entities = {}
        for object_type, objects in {"Area": areas, "Output": outputs}.items():
            for obj in objects:
                current_obj = obj.value
                object_id = obj.id
                id_record = self._id_map.get(obj.id)
                if id_record is None:
                    raise ValueError(f"No ID record found for: {obj.id}")
                parent_id = id_record.parent
                path = self._build_path(obj)
                current_obj = self._applyFilters(current_obj)
                spoken_name = self._constructSpokenName(current_obj)
                subtype = current_obj['type'] if 'type' in current_obj else None
                entities[object_id] = LutronDBEntity(
                    object_id,
                    current_obj['iid'],
                    current_obj['name'],
                    spoken_name,
                    object_type,
                    subtype,
                    self._path_to_string(path),
                    parent_id,
                )

        self._entities = entities

    def _build_path(self, obj: ObjectHolder) -> list[dict]:
        path = []

        record: IIDMapRecord | None = self._id_map.get(obj.id)
        while record is not None:
            path.insert(0, record.object.value)

            parent_id = record.parent
            if parent_id is None:
                break
            record = self._id_map.get(parent_id)
        
        return path

    def _path_to_string(self, path: list[dict]):
        return " / ".join([self._constructSpokenName(obj) for obj in path])
    
    def _applyFilters(self, data: dict):
        for filter in self._filters:
            data = filter(data)
        return data
    
    def _constructSpokenName(self, data: dict):
        name = data['name']

        # Remove leading digits
        name = re.sub(r'^\d+', '', name)

        # Remove trailing digits
        name = re.sub(r'\d+$', '', name)

        # Remove duplicate spaces
        name = re.sub(r'\s+', ' ', name)

        # Remove leading and trailing whitespace
        name = name.strip()

        return name
    
    def _mapType(self, entity: LutronDBEntity):
        # TODO: Index type_map by type for faster lookup
        for result_type, type_list in self.config.get("tools.lutron.type_map").items():
            if entity.subtype in type_list:
                return result_type
        return None
    
    def getEntities(self) -> list[LutronEntity]:
        result = []
        for entity in self._entities.values():
            iid = entity.iid
            name = entity.name
            spoken_name = entity.spoken_name or entity.name
            path = entity.path
            output_type = self._mapType(entity)

            # Filter out unmapped output types
            if entity.type == "Output" and output_type is None:
                continue

            result.append(LutronEntity(iid, name, spoken_name, entity.type, output_type, path))
        return result
    
    def getAreas(self, include_parents=False):
        result = []
        for area in self.database.getAreas():
            if area['is_leaf'] or include_parents:
                area = self._applyFilters(area)
                spoken_name = self._constructSpokenName(area)
                if area['is_leaf']:
                    iid = area['iid']
                    obj = LutronArea(iid, spoken_name)
                    result.append(obj)
                else:
                    obj = LutronAreaGroup(spoken_name)
                    result.append(obj)
        return result

    def getOutputs(self) -> list[LutronOutput]:
        results = []
        
        for entity in self._entities.values():
            if entity.type == "Output":
                iid = entity.iid
                spoken_name = entity.spoken_name or entity.name
                path = entity.path
                output_type = self._mapType(entity)

                # Filter out unmapped output types
                if output_type is None:
                    continue

                obj = LutronOutput(iid, output_type, spoken_name, path)
                results.append(obj)
        return results


class Filter:
    def __init_subclass__(cls, filter_name: str, **kwargs):
        cls.filter_name = filter_name

        LutronDatabaseView.register_filter(cls)
        super().__init_subclass__(**kwargs)

class NameReplaceFilter(Filter, filter_name='name_replace'):

    def __init__(self, old_fragment: str, new_fragment: str):
        self.old_fragment = old_fragment
        self.new_fragment = new_fragment
    
    def __call__(self, data: dict) -> dict:
        data['name'] = data['name'].replace(self.old_fragment, self.new_fragment)
        return data

class PreserveNumberFilter(Filter, filter_name='preserve_number'):
    known_numbers = {
        '0': 'Zero',
        '1': 'One',
        '2': 'Two',
        '3': 'Three',
        '4': 'Four',
        '5': 'Five',
        '6': 'Six',
        '7': 'Seven',
        '8': 'Eight',
        '9': 'Nine',
    }

    def __init__(self, name_match: str):
        self.name_match = name_match
    
    @classmethod
    def number_replacer(cls, match: re.Match) -> str:
        number = match.group(0)
        return cls.known_numbers[number] if number in cls.known_numbers else number
    
    def __call__(self, data: dict) -> dict:
        assert isinstance(data['name'], str)
        if self.name_match in data['name']:
            # Find the number in the name and replace with lookup table
            data['name'] = re.sub(r'\d+', self.number_replacer, data['name'])
        return data

class TypeFixFilter(Filter, filter_name='type_fix'):
    def __init__(self, name_match: str, type: str):
        self.name_match = name_match
        self.type = type
    
    def __call__(self, data: dict) -> dict:
        if self.name_match in data['name']:
            data['type'] = self.type
        return data
        