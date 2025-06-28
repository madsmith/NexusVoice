from dataclasses import dataclass, field
import logging
import re

from nexusvoice.core.config import NexusConfig

from .database import LutronDatabase

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
    name: str

    @classmethod
    def from_dict(cls, data):
        return cls(data['iid'], data['name'])

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

    def initialize(self):
        # Initialize filters based on config and available filters
        for filter_type, filter_rules in self.config.get("tools.lutron.filters").items():
            filter_class = self._available_filters.get(filter_type)
            if filter_class is None:
                self.logger.warning(f"Unknown filter type: {filter_type}")
                continue
            
            for filter_rule in filter_rules:
                self._filters.append(filter_class(*filter_rule))
        return self

    def getEntities(self):
        db_iid_map = self.database.getIIDMap()
        areas = [ObjectHolder(area) for area in self.database.getAreas()]
        outputs = [ObjectHolder(output) for output in self.database.getOutputs()]

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
        
        # Rebuild iid_map on DataHolder using each objects DataHolder.id
        iid_map = {}
        print("Building new iid_map")
        for record in db_iid_map.values():
            obj_type = record['type']
            if obj_type not in self.config.get("tools.lutron.valid_object_types"):
                print(f"Skipping invalid type: {obj_type}")
                continue
            print(record)
            obj = find_object(record['iid'], record['name'])
            if obj is None:
                raise ValueError(f"No object found for: {record['iid']} {record['name']} {record['type']}")
            parent = None
            if record['parent_iid'] is not None or record['parent_name'] is not None:
                parent = find_object(record['parent_iid'], record['parent_name'])
                if parent is None:
                    raise ValueError(f"No parent found for: {record['parent_iid']} {record['parent_name']} {record['parent_type']}")
            else:
                print(f"No parent found for: {record['iid']} {record['name']} {record['type']}")

            parent_id = parent.id if parent is not None else None
            print(f"Mapping {obj.value['name']} ({obj.value['iid']}) @ {obj.id} to {parent_id}")
            iid_map[obj.id] = {
                "object": obj,
                "parent": parent_id
            }

        entities = []
        for object_type, objects in {"Area": areas, "Output": outputs}.items():
            for obj in objects:
                current_obj = obj.value
                print("Current", current_obj)
                path = self._build_path(obj, iid_map)
                current_obj = self._applyFilters(current_obj)
                spoken_name = self._constructSpokenName(current_obj)
                subtype = current_obj['subtype'] if 'subtype' in current_obj else None
                entities.append(LutronEntity(
                    current_obj['iid'],
                    current_obj['name'],
                    spoken_name,
                    object_type,
                    subtype,
                    self._path_to_string(path),
                ))
        return entities

    def _build_path(self, obj: ObjectHolder, iid_map: dict[int, dict]) -> list[dict]:
        print(f"Building path for {obj.value['name']} ({obj.value['iid']})")

        path = []

        record: dict | None = iid_map.get(obj.id)
        while record is not None:
            print("  Record", record)
            path.insert(0, record['object'].value)

            parent_id = record['parent']
            if parent_id is None:
                break
            record = iid_map.get(parent_id)
        
        return path

    def _path_to_string(self, path: list[dict]):
        return " / ".join([self._constructSpokenName(obj) for obj in path])
    
    def _applyFilters(self, data: dict):
        print(f"Applying filters to {data['name']} ({data['iid']})")
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
        
    def getAreas(self, include_parents=False):
        areas = self.database.getAreas()
        filtered_areas = self.filterAreas(areas, include_parents)
        return filtered_areas

    def filterAreas(self, areas, include_parents=False):
        results = []
        for area in areas:
            if area['is_leaf'] or include_parents:
                area = self._applyFilters(area)
                obj = LutronArea.from_dict(area)
                results.append(obj)
        return results

    def getOutputs(self):
        outputs = self.database.getOutputs()

        filtered_outputs = self.filterOutputs(outputs)
        return filtered_outputs

    def filterOutputs(self, outputs):
        results = []
        for output in outputs:
            output = self._applyFilters(output)
            obj = LutronOutput.from_dict(output)
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