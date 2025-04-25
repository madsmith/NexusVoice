from typing import Any
from omegaconf import OmegaConf, DictConfig, ListConfig

class NexusConfig:
    def __init__(self, config: DictConfig | ListConfig):
        self._config = config

    def get(self, key: str, default: Any = None) -> Any:
        return OmegaConf.select(self._config, key, default=default)

    def set(self, key: str, value: Any) -> None:
        def highlight_key(parts, highlight_idx):
            return '.'.join([
                p if j != highlight_idx else f'**{p}**'
                for j, p in enumerate(parts)
            ])

        key_parts = key.split('.')

        current_config = self._config
        for i, part in enumerate(key_parts[:-1]):
            if isinstance(current_config, DictConfig):
                if part not in current_config:
                    current_config[part] = {}
                current_config = current_config[part]
            elif isinstance(current_config, ListConfig):
                try:
                    index = int(part)
                    if index < 0 or index >= len(current_config):
                        highlighted = highlight_key(key_parts, i)
                        raise ValueError(f"Index {index} out of bounds in \"{highlighted}\"")
                    current_config = current_config[index]
                except ValueError:
                    highlighted = highlight_key(key_parts, i)
                    raise ValueError(f"Invalid list index: {part} in \"{highlighted}\"")
            else:
                highlighted = highlight_key(key_parts, i)
                raise ValueError(f"Config must be a DictConfig or ListConfig at \"{highlighted}\"")

        last_part = key_parts[-1]
        if isinstance(current_config, DictConfig):
            current_config[last_part] = value
        elif isinstance(current_config, ListConfig):
            try:
                index = int(last_part)
                if index < 0 or index >= len(current_config):
                    highlighted = highlight_key(key_parts, len(key_parts)-1)
                    raise ValueError(f"Index {index} out of bounds in \"{highlighted}\"")
                current_config[index] = value
            except ValueError:
                highlighted = highlight_key(key_parts, len(key_parts)-1)
                raise ValueError(f"Invalid list index: {last_part} in \"{highlighted}\"")
        else:
            highlighted = highlight_key(key_parts, len(key_parts)-1)
            raise ValueError(f"Config must be a DictConfig or ListConfig at \"{highlighted}\"")

    def __getattr__(self, name: str) -> Any:
        return self.get(name)

def load_config() -> NexusConfig:
    config = OmegaConf.load("config.yml")
    return NexusConfig(config)