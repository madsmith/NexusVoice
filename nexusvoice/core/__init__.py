"""
Nexus Core subpackage - A portable subset of NexusVoice core functionality.
This package is designed to be used by other applications without requiring the full NexusVoice codebase.
"""

from .config import NexusConfig, load_config

__all__ = ["NexusConfig", "load_config"]
