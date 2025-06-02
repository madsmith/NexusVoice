import shlex
from pydantic_ai import Agent, RunContext

from nexusvoice.ai.ConversationalAgent import ConversationalAgentFactory
from nexusvoice.ai.HomeAutomationAgent import HomeAutomationAgentFactory
from nexusvoice.ai.LocalClassifierAgent import LocalClassifierAgentFactory
from nexusvoice.ai.types import (
    ConversationResponse,
    HomeAutomationResponse,
    HomeAutomationResponseStruct,
    NexusSupportDependencies,
    RequestType
)
from nexusvoice.core.api import NexusAPI
from nexusvoice.core.config import NexusConfig
from nexusvoice.utils.logging import get_logger
from pydantic_ai.mcp import MCPServerStdio

logger = get_logger(__name__)

from nexusvoice.tools.weather import get_weather

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

        self._mcp_servers = {}

        self._classifier_agent = None
        self._home_agent = None
        self._conversational_agent = None


    async def initialize(self):
        logger.debug("Initializing agents...")
        self._deps = NexusSupportDependencies(config=self.config)

        # Initialize MCP servers
        await self._initialize_mcp_servers()

        self._deps.servers = list(self._mcp_servers.values())

        self._classifier_agent = LocalClassifierAgentFactory.create(self._deps)
        self._home_agent = HomeAutomationAgentFactory.create(self._deps)
        self._conversational_agent = ConversationalAgentFactory.create(self._deps)

        # Register tools
        self._home_agent.tool(home_control)
        self._conversational_agent.tool(get_weather)


    async def _initialize_mcp_servers(self):
        logger.debug("Initializing MCP servers...")
        servers = self.config.get("servers", [])
        for server in servers:
            await self._create_mcp_server(server)

    async def _create_mcp_server(self, server: dict):
        logger.debug(f"Creating MCP server {server['name']}...")
        
        if server["transport"] == "stdio":
            await self._create_stdio_mcp_server(server)
        else:
            logger.warning(f"Unknown transport type {server['transport']}")

    async def _create_stdio_mcp_server(self, server: dict):
        if 'command' not in server:
            logger.warning("Missing command for stdio server")
            return

        args = []
        if 'args' in server:
            if isinstance(server["args"], list):
                args = server["args"]
            else:
                args = shlex.split(server["args"])
        
        prefix = server.get("prefix", "tool")
        in_use_count = 1
        while prefix in self._mcp_servers.keys():
            prefix += f"_{in_use_count}"
            in_use_count += 1
        
        instance = MCPServerStdio(
            server["command"],
            args=args,
            tool_prefix=prefix
        )

        self._mcp_servers[prefix] = instance
        
    @property
    def classifier_agent(self) -> Agent[NexusSupportDependencies, RequestType]:
        assert self._classifier_agent is not None, "Classifier agent not initialized"
        return self._classifier_agent
    
    @property
    def home_agent(self) -> Agent[NexusSupportDependencies, HomeAutomationResponse]:
        assert self._home_agent is not None, "Home automation agent not initialized"
        return self._home_agent
    
    @property
    def conversational_agent(self) -> Agent[NexusSupportDependencies, ConversationResponse]:
        assert self._conversational_agent is not None, "Conversational agent not initialized"
        return self._conversational_agent

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

    async def _process_home_automation(self, text: str) -> str | None:
        logger.debug("Processing home automation request...")
        try:
            result = await self.home_agent.run(text, deps=self._deps)
            
            message = HomeAutomationResponseStruct.extract_message(result.output)
            return message
        except Exception as e:
            logger.debug(f"Home automation processing failed: {e}")
            
        return None

    async def _process_conversational(self, text: str) -> str:
        logger.debug("Processing conversational request...")
        result = await self.conversational_agent.run(text, deps=self._deps)
        return result.output.text
