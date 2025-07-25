import logfire
import logging
from pydantic_ai.mcp import MCPServer, MCPServerStdio
import shlex

logger = logging.getLogger(__name__)

class Prefix:
    def __init__(self, prefix):
        self._prefix = prefix
        self._instance_id: int | None = None

    def with_instance_id(self, instance_id: int) -> "Prefix":
        self._instance_id = instance_id
        return self

    @property
    def base_prefix(self) -> str:
        return self._prefix
    
    @property
    def prefix(self) -> str:
        suffix = ""
        if self._instance_id is not None and self._instance_id > 1:
            suffix = f"_{self._instance_id}"
        return f"{self._prefix}{suffix}"

class PrefixNamespace:
    def __init__(self):
        self._prefix_counts = {}

    def new(self, prefix: str) -> Prefix:
        return (
            Prefix(prefix)
            .with_instance_id(self._allocate_prefix(prefix))
        )

    def _allocate_prefix(self, prefix: str) -> int:
        if prefix in self._prefix_counts:
            self._prefix_counts[prefix] += 1
        else:
            self._prefix_counts[prefix] = 1
        
        return self._prefix_counts[prefix]

class MCPConfigFactory():
    def __init__(self):
        self._prefix_ns = PrefixNamespace()
    
    def create(self, config: dict) -> MCPServer:
        if config["transport"] == "stdio":
            return self._create_stdio_mcp_server(config)
        else:
            logger.warning(f"Unknown transport type {config['transport']}")
    
    def _create_stdio_mcp_server(self, config: dict) -> MCPServerStdio:
        if 'command' not in config:
            logger.error("Missing command for stdio server")
            raise ValueError("Missing command for stdio server")

        # Parse args
        args = []
        if 'args' in config:
            if isinstance(config["args"], list):
                args = config["args"]
            else:
                args = shlex.split(config["args"])

        # Parse environment variables
        env = {}
        if 'env' in config:
            env = config["env"]

        # Parse prefix - ensure each server has a unique prefix
        base_prefix = config.get("prefix", "tool")
        prefix = self._prefix_ns.new(base_prefix)
        
        return MCPServerStdio(
            config["command"],
            args=args,
            tool_prefix=prefix.prefix,
            env=env
        )
        
        