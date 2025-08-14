# NexusVoice Core

This subpackage provides essential, reusable components from NexusVoice that can be used independently in other projects without requiring the full NexusVoice codebase.

## Components

### NexusConfig

A flexible configuration management class that works with OmegaConf.

```python
from nexusvoice.core import NexusConfig
from omegaconf import OmegaConf

# Create a config from an OmegaConf object
config_dict = {"server": {"host": "localhost", "port": 8080}}
omega_config = OmegaConf.create(config_dict)
config = NexusConfig(omega_config)

# Access configuration
host = config.get("server.host")  # returns "localhost"
port = config.server.port  # returns 8080

# Set configuration
config.set("server.debug", True)
config.server.timeout = 30
```

### load_config()

A utility function that loads configuration from YAML files:

```python
from nexusvoice.core import load_config

# Loads config from config/config.yaml and config/config_private.yaml
# and merges them
config = load_config()
```

## Installation Options

### Option 1: Install from GitHub (Recommended)

```bash
# Install directly from GitHub repository
pip install git+https://github.com/madsmith/NexusVoiceRedux.git#subdirectory=nexusvoice/core
```

### Option 2: Add as a Dependency in Your Project

In your project's `pyproject.toml`:

```toml
[tool.poetry.dependencies]
nexusvoice-core = { git = "https://github.com/madsmith/NexusVoiceRedux.git", subdirectory = "nexusvoice/core" }
```

### Option 3: Install from Local Source

If you have a local copy of the NexusVoice repository:

```bash
# Navigate to the repository
cd /path/to/NexusVoiceRedux

# Install in development mode (editable)
pip install -e .
```

## Requirements

- Python 3.10+
- omegaconf >= 2.3.0
