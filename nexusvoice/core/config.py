from typing import Any
from omegaconf import OmegaConf, DictConfig, ListConfig

class NexusConfig:
    def __init__(self, config: DictConfig | ListConfig):
        self._config = config

    def get(self, key: str, default: Any = None) -> Any:
        return OmegaConf.select(self._config, key, default=default)

    def __getattr__(self, name: str) -> Any:
        return self.get(name)


def load_config() -> NexusConfig:
    config = OmegaConf.load("config.yml")
    return NexusConfig(config)