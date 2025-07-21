import asyncio
from contextlib import AsyncExitStack
import logfire
import shlex
from typing import Optional
from nexusvoice.core.api.base import NexusHistoryContext
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

from nexusvoice.ai.ConversationalAgent import ConversationalAgentFactory
from nexusvoice.ai.HomeAutomationAgent import HomeAutomationAgentFactory
from nexusvoice.ai.LocalClassifierAgent import LocalClassifierAgentFactory
from nexusvoice.ai.types import (
    ConversationResponse,
    HomeAutomationResponseStruct,
    NexusSupportDependencies,
    RequestType
)
from nexusvoice.core.api import NexusAPI, NexusAPIContext

from nexusvoice.core.config import NexusConfig
from nexusvoice.utils.logging import get_logger
from nexusvoice.utils.debug import DebugContext
from pydantic_ai.mcp import MCPServer, MCPServerStdio

logger = get_logger(__name__)

from nexusvoice.tools.weather import get_weather

class Prefix:
    # Class variable to track instances of each prefix
    _prefix_counts = {}
    
    def __init__(self, prefix):
        self._prefix = prefix
        
        # Update the reference count and set instance_id
        self._instance_id = Prefix.allocate_prefix(prefix)

    @property
    def base_prefix(self) -> str:
        return self._prefix
    
    @property
    def prefix(self) -> str:
        suffix = ""
        if self._instance_id > 1:
            suffix = f"_{self._instance_id}"
        return f"{self._prefix}{suffix}"
    
    @classmethod
    def allocate_prefix(cls, prefix):
        """Allocate a new prefix for a given prefix"""
        if prefix in cls._prefix_counts:
            cls._prefix_counts[prefix] += 1
        else:
            cls._prefix_counts[prefix] = 1
        
        return cls._prefix_counts[prefix]

class NexusOnlineAPIContext(NexusAPIContext["NexusOnlineAPIContext"]):
    def __init__(self, api: "NexusAPIOnline"):
        self.api = api
        self._is_open = False
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "NexusOnlineAPIContext":
        await self._stack.__aenter__()

        agents = [
            self.api._home_agent,
            self.api._classifier_agent,
            self.api._conversational_agent,
        ]
        
        for i, agent in enumerate(agents):
            # TODO: consider if mcp servers should have a life context separate
            # from the api context
            ctx = agent.run_mcp_servers()
            await self._stack.enter_async_context(ctx)
            # await self._stack.enter_async_context(DebugContext(ctx, f"Agent {i}", logger.debug))

        self._is_open = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_open = False
        try:
            await self._stack.__aexit__(exc_type, exc_val, exc_tb)
        except asyncio.CancelledError:
            pass

    def is_open(self) -> bool:
        return self._is_open

class NexusOnlineHistoryContext(NexusHistoryContext["NexusOnlineHistoryContext"]):
    def __init__(self, agent_history: list[ModelMessage]):
        self.agent_history = agent_history
    
    async def __aenter__(self) -> "NexusOnlineHistoryContext":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

#==========================================================
# Tools
#==========================================================

def home_control(ctx: RunContext[NexusSupportDependencies], intent: str, device: str, room: str) -> dict[str, str]:
    """
    Turn on, off, raise, or lower a home automation device.
    
    Args:
        ctx: The run context
        intent: The action to perform (e.g., turn_on, turn_off, raise, lower)
        device: The device to control (e.g., light, fan, shade)
        room: The room where the device is located
    """
    status = {
        "result": f"The {device} in the {room} has been updated.  Status: {intent}",
        "intent": intent,
        "device": device,
        "room": room
    }
    print(f"Home Automation: {intent} {device} in {room}")
    return {
        "name": "home_control",
        "result": "Action completed"
    }
    
class NexusAPIOnline(NexusAPI):
    def __init__(self, config: NexusConfig):
        super().__init__(config)

        self._mcp_servers: dict[str, MCPServer] = {}

        self._classifier_agent = None
        self._home_agent = None
        self._conversational_agent = None

        self._context: Optional[NexusOnlineAPIContext] = None
        self._history_context: Optional[NexusOnlineHistoryContext] = None


    async def initialize(self):
        logger.debug("Initializing agents...")
        self._deps = NexusSupportDependencies(config=self.config)

        # Initialize MCP servers
        await self._initialize_mcp_servers()

        self._deps.servers = self._mcp_servers

        self._classifier_agent = LocalClassifierAgentFactory.create(self._deps)
        self._home_agent = HomeAutomationAgentFactory.create(self._deps)
        self._conversational_agent = ConversationalAgentFactory.create(self._deps)

        # Register tools
        # self._home_agent.tool(home_control)
        self._conversational_agent.tool(get_weather)

    async def _initialize_mcp_servers(self):
        logger.debug("Initializing MCP servers...")
        server_configs = self.config.get("mcp-server-configs", [])
        for server_config in server_configs:
            await self._create_mcp_server(server_config)

    async def _create_mcp_server(self, server_config: dict):
        logger.debug(f"Creating MCP server {server_config['name']}...")
        
        if server_config["transport"] == "stdio":
            await self._create_stdio_mcp_server(server_config)
        else:
            logger.warning(f"Unknown transport type {server_config['transport']}")

    async def _create_stdio_mcp_server(self, server_config: dict):
        if 'command' not in server_config:
            logger.error("Missing command for stdio server")
            return

        # Parse args
        args = []
        if 'args' in server_config:
            if isinstance(server_config["args"], list):
                args = server_config["args"]
            else:
                args = shlex.split(server_config["args"])

        # Parse environment variables
        env = {}
        if 'env' in server_config:
            env = server_config["env"]
        
        name = server_config['name']

        # Parse prefix - ensure each server has a unique prefix
        base_prefix = server_config.get("prefix", "tool")
        prefix = Prefix(base_prefix)
        
        instance = MCPServerStdio(
            server_config["command"],
            args=args,
            tool_prefix=prefix.prefix,
            env=env
        )

        self._mcp_servers[name] = instance

    async def create_api_context(self) -> NexusOnlineAPIContext:
        """
        Returns an async context manager for a NexusAPIOnline session.
        Initializes MCP servers and attaches them to the context.
        """
        ctx = NexusOnlineAPIContext(
            api=self
        )

        # Check if the context is already initialized and open, and warn of 
        # possible errors if that's the case
        if self._context is not None and self._context.is_open():
            logger.warning("Context is already initialized and open.")
        
        # Set the context so that it can be accessed later
        self._context = ctx
        return ctx

    async def create_history_context(self) -> NexusOnlineHistoryContext:
        """
        Returns an async context manager for a NexusOnlineHistory session.
        """
        agent_history = []
        ctx = NexusOnlineHistoryContext(agent_history=agent_history)
        self._history_context = ctx
        return ctx

    @property
    def api_context(self) -> NexusOnlineAPIContext:
        """
        Returns the current context for the API session.
        It's expected that the API user is managing the context lifecycle and has called create_api_context() to initialize it.
        """
        assert self._context is not None, "Context not initialized"
        return self._context

    @property
    def history_context(self) -> NexusOnlineHistoryContext:
        assert self._history_context is not None, "History context not initialized"
        return self._history_context

    @property
    def classifier_agent(self) -> Agent[NexusSupportDependencies, RequestType]:
        assert self._classifier_agent is not None, "Classifier agent not initialized"
        return self._classifier_agent
    
    @property
    def home_agent(self) -> Agent[NexusSupportDependencies, HomeAutomationResponseStruct]:
        assert self._home_agent is not None, "Home automation agent not initialized"
        return self._home_agent
    
    @property
    def conversational_agent(self) -> Agent[NexusSupportDependencies, ConversationResponse]:
        assert self._conversational_agent is not None, "Conversational agent not initialized"
        return self._conversational_agent

    @logfire.instrument("Prompt Agent: {agent_id}")
    async def prompt_agent(self, agent_id: str, prompt: str) -> str:
        # First use the classifier to determine request type
        # Try local classifier first
        classification = await self._classify_request(prompt)

        logger.debug(f"Request classified as {classification.type} with confidence {classification.confidence}")

        # If confident it's a home automation request, use the home automation agent
        if classification.type == "home_automation" and classification.confidence > 0.7:
            response = await self._process_home_automation(prompt)
            
            return response if response else ""

        # Fall back to conversational response
        return await self._process_conversational(prompt)

    @logfire.instrument("Classify Request")
    async def _classify_request(self, text: str) -> RequestType:
        try:
            result = await self.classifier_agent.run(text, deps=self._deps)
            return result.output
        except Exception as e:
            logger.warning(f"Local classifier failed: {e}, defaulting to conversation")
            # print stack trace
            import traceback
            traceback.print_exc()
            return RequestType(
                type="conversation",
                confidence=0.0
            )

    @logfire.instrument("Process Home Automation")
    async def _process_home_automation(self, text: str) -> str | None:
        logger.debug("Processing home automation request...")
        try:
            result = await self.home_agent.run(text, deps=self._deps)
            
            message = HomeAutomationResponseStruct.extract_message(result.output)
            return message
        except Exception as e:
            logger.debug(f"Home automation processing failed: {e}")
            
        return None

    @logfire.instrument("Process Conversational")
    async def _process_conversational(self, text: str) -> str:
        logger.debug("Processing conversational request...")
        message_history = self.history_context.agent_history

        result = await self.conversational_agent.run(
            text,
            message_history=message_history,
            deps=self._deps)

        # Update the agent history
        self.history_context.agent_history = result.all_messages()

        return result.output.text
