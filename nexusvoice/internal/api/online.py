import anyio
import asyncio
from contextlib import AsyncExitStack
import logfire
import shlex
from typing import Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, TextPart

from nexusvoice.ai.ConversationalAgent import ConversationalAgentFactory
from nexusvoice.ai.HomeAutomationAgent import HomeAutomationAgentFactory
from nexusvoice.ai.LocalClassifierAgent import LocalClassifierAgentFactory
from nexusvoice.ai.types import (
    ConversationResponse,
    HomeAutomationResponseStruct,
    NexusSupportDependencies,
    RequestType
)
from nexusvoice.internal.api import NexusAPI, NexusAPIContext
from nexusvoice.internal.api.base import NexusHistoryContext

from nexusvoice.core.config import NexusConfig
from nexusvoice.utils.logging import get_logger
from nexusvoice.utils.debug import DebugContext
from pydantic_ai.mcp import MCPServer, MCPServerStdio
from pydantic_ai.models import ModelResponse

logger = get_logger(__name__)

from nexusvoice.tools.weather import get_weather

from .pydantic_mcp import MCPConfigFactory


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

        self._classifier_agent: Agent[NexusSupportDependencies, RequestType] | None = None
        self._home_agent: Agent[NexusSupportDependencies, HomeAutomationResponseStruct] | None = None
        self._conversational_agent: Agent[NexusSupportDependencies, ConversationResponse] | None = None

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
        factory = MCPConfigFactory()
        server_configs = self.config.get("mcp-server-configs", [])
        for server_config in server_configs:
            name = server_config["name"]
            mcp_server = factory.create(server_config)
            if mcp_server is None:
                raise ValueError(f"Failed to create MCP server for {name}")
            self._mcp_servers[name] = mcp_server

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
    async def _process_home_automation(self, text: str) -> str:
        logger.debug("Processing home automation request...")
        try:
            with logfire.span("Home automation agent run"):
                with anyio.fail_after(float(self.config.get("nexus.client.timeouts.agent_run", 30))):
                    result = await self.home_agent.run(text, deps=self._deps)
            
            message = HomeAutomationResponseStruct.extract_message(result.output)
            return message
        except TimeoutError:
            return "The previous request took too long to process and was cancelled."
        except Exception as e:
            logger.debug(f"Home automation processing failed: {e}")
            
        return "The previous request resulted in an unexpected error."

    @logfire.instrument("Process Conversational")
    async def _process_conversational(self, text: str) -> str:
        logger.debug("Processing conversational request...")
        message_history = self.history_context.agent_history

        try:
            with logfire.span("Conversational agent run"):
                with anyio.fail_after(float(self.config.get("nexus.client.timeouts.agent_run", 30))):
                    result = await self.conversational_agent.run(
                        text,
                        message_history=message_history,
                        deps=self._deps)
    
            # Update the agent history
            self.history_context.agent_history = result.all_messages()

            return result.output.text
        except (TimeoutError, anyio.ClosedResourceError):
            return "The previous request took too long to process and was cancelled."
        except BaseException as e:
            logger.exception("Exception in process_conversational")
            self.history_context.agent_history.append(
                ModelResponse(
                    parts = [TextPart(content="The previous request resulted in an unexpected error.")]
                )
            )
            return "I'm sorry, I was unable to process your request."

