import pytest
from omegaconf import OmegaConf
from nexusvoice.core.config import NexusConfig

@pytest.fixture
def simple_config():
    return OmegaConf.create({
        'foo': 'bar',
        'number': 42,
        'nested': {
            'key1': 'value1',
            'key2': 123,
        },
    })

@pytest.fixture
def config_with_defaults():
    return OmegaConf.create({
        'foo': 'bar',
        'nested': {
            'key1': 'value1',
        },
    })

def list_config_fixture():
    # Structure: root is dict, contains a list, which contains dicts, which contain lists, etc.
    return OmegaConf.create({
        'toplist': [
            {'innerdict': {'x': 1, 'y': [10, 20, 30]}},
            {'innerdict': {'x': 2, 'y': [40, 50]}}
        ],
        'dictlist': {
            'alist': [100, 200, {'foo': 'bar'}],
            'nested': [
                {'a': [1, 2]},
                {'b': [3, 4]}
            ]
        }
    })

def test_get_top_level_key(simple_config):
    cfg = NexusConfig(simple_config)
    assert cfg.get('foo') == 'bar'
    assert cfg.get('number') == 42

def test_get_nested_key(simple_config):
    cfg = NexusConfig(simple_config)
    assert cfg.get('nested.key1') == 'value1'
    assert cfg.get('nested.key2') == 123

def test_missing_key_returns_none(simple_config):
    cfg = NexusConfig(simple_config)
    assert cfg.get('does_not_exist') is None
    assert cfg.get('nested.nope') is None

def test_missing_key_with_default(simple_config):
    cfg = NexusConfig(simple_config)
    assert cfg.get('does_not_exist', 'default') == 'default'
    assert cfg.get('nested.nope', 999) == 999

def test_getattr_access(simple_config):
    cfg = NexusConfig(simple_config)
    assert cfg.foo == 'bar'
    assert cfg.number == 42

def test_getattr_missing_key(simple_config):
    cfg = NexusConfig(simple_config)
    assert cfg.get('not_a_key') is None

def test_nested_missing_key_with_default(config_with_defaults):
    cfg = NexusConfig(config_with_defaults)
    assert cfg.get('nested.key2', 'fallback') == 'fallback'

def test_set_top_level_key(simple_config):
    cfg = NexusConfig(simple_config)
    cfg.set('new_key', 'new_value')
    assert cfg.get('new_key') == 'new_value'

def test_set_nested_key(simple_config):
    cfg = NexusConfig(simple_config)
    cfg.set('nested.key3', 'val3')
    assert cfg.get('nested.key3') == 'val3'

def test_overwrite_existing_key(simple_config):
    cfg = NexusConfig(simple_config)
    cfg.set('foo', 'baz')
    assert cfg.get('foo') == 'baz'

def test_set_new_nested_key(simple_config):
    cfg = NexusConfig(simple_config)
    cfg.set('newparent.child', 123)
    assert cfg.get('newparent.child') == 123

def test_set_list_index():
    cfg = NexusConfig(list_config_fixture())
    # Set a primitive in a list
    cfg.set('dictlist.alist.0', 111)
    assert cfg.get('dictlist.alist.0') == 111
    # Set a dict in a list
    cfg.set('dictlist.alist.2.foo', 'baz')
    assert cfg.get('dictlist.alist.2.foo') == 'baz'

def test_set_nested_list_in_dict():
    cfg = NexusConfig(list_config_fixture())
    # Set a value in a nested list inside a dict inside a list
    cfg.set('toplist.0.innerdict.y.1', 77)
    assert cfg.get('toplist.0.innerdict.y.1') == 77
    # Set a new value in a nested list
    cfg.set('toplist.1.innerdict.y.1', 88)
    assert cfg.get('toplist.1.innerdict.y.1') == 88

def test_set_dict_in_list_of_dicts():
    cfg = NexusConfig(list_config_fixture())
    # Set a dict key inside a dict inside a list
    cfg.set('toplist.1.innerdict.z', 999)
    assert cfg.get('toplist.1.innerdict.z') == 999

def test_set_list_in_list_of_dicts():
    cfg = NexusConfig(list_config_fixture())
    # Set a new list inside a dict inside a list
    cfg.set('toplist.0.innerdict.newlist', [1, 2, 3])
    assert cfg.get('toplist.0.innerdict.newlist.2') == 3

def test_alternating_dict_list_set():
    cfg = NexusConfig(list_config_fixture())
    # Dict -> List -> Dict -> List
    cfg.set('dictlist.nested.1.b.0', 42)
    assert cfg.get('dictlist.nested.1.b.0') == 42
    # List -> Dict -> List
    cfg.set('toplist.0.innerdict.y.2', 12345)
    assert cfg.get('toplist.0.innerdict.y.2') == 12345
