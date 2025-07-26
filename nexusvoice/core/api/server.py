import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
import logfire
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServer
from pydantic_ai.messages import ModelMessage

from nexusvoice.ai.ConversationalAgent import ConversationalAgentFactory
from nexusvoice.ai.HomeAutomationAgent import HomeAutomationAgentFactory
from nexusvoice.ai.LocalClassifierAgent import LocalClassifierAgentFactory
from nexusvoice.ai.types import NexusSupportDependencies, RequestType
from nexusvoice.core.config import NexusConfig
from nexusvoice.utils.RuntimeContextManager import RuntimeContextManager

from .base import NexusHistoryContext
from .pydantic_mcp import MCPConfigFactory

# TODO: figure out what T type to use here... at the moment we're not
# calling "async with NexusServerHistoryContext()" so it's doesn't have
# a relevant access pattern
class NexusServerHistoryContext(NexusHistoryContext["NexusServerHistoryContext"]):
    def __init__(self, agent_history: list[ModelMessage]):
        self.agent_history = agent_history

    async def __aenter__(self) -> "NexusServerHistoryContext":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class NexusClientContext(NexusSupportDependencies):
    def __init__(
        self,
        config: NexusConfig,
        servers: dict[str, MCPServer],
        history: list[ModelMessage]
    ):
        self.config = config
        self.servers = servers
        self.history = history
    
class NexusAPIServer():
    def __init__(self, config: NexusConfig):
        self._config = config

        self._context_manager = RuntimeContextManager()

        self._client_ids: list[str] = []

        self._mcp_servers: dict[str, MCPServer] = {}
        
        self._classifier_agent: Agent[NexusSupportDependencies, RequestType] | None = None
        self._home_agent: Agent[NexusSupportDependencies, RequestType] | None = None
        self._conversational_agent: Agent[NexusSupportDependencies, RequestType] | None = None

        self._support_deps: NexusSupportDependencies | None = None

        self._client_history_contexts: dict[str, NexusHistoryContext] = {}

    @logfire.instrument("NexusAPIServer.initialize")
    async def initialize(self):
        # Create MCP Server Objects for agents
        mcp_configs = self._config.get("mcp-server-configs", [])
        factory = MCPConfigFactory()
        for server_config in mcp_configs:
            name = server_config["name"]
            mcp_server = factory.create(server_config)
            if mcp_server is None:
                raise ValueError(f"Failed to create MCP server for {name}")
            self._mcp_servers[name] = mcp_server

        # Add context for agents
        self._context_manager.add_context(
            "agent-context",
            self.nexus_online_api_context,
            self._config.get("nexus.server.timeouts.agent_context", 300)
        )

        self._support_deps = NexusSupportDependencies(
            config=self._config,
            servers=self._mcp_servers # TODO: remove?
        )

        # Initialize Agents
        self._classifier_agent = LocalClassifierAgentFactory.create(self._support_deps)
        self._home_agent = HomeAutomationAgentFactory.create(self._support_deps)
        self._conversational_agent = ConversationalAgentFactory.create(self._support_deps)

        self._context_manager.start()

    @asynccontextmanager
    async def nexus_online_api_context(self):
        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            agents = [
                self._home_agent,
                self._classifier_agent,
                self._conversational_agent,
            ]

            for i, agent in enumerate(agents):
                assert agent is not None, f"Agent {i} is not initialized"
                await stack.enter_async_context(agent.run_mcp_servers())

            yield 
        except asyncio.CancelledError:
            pass
        finally:
            await stack.__aexit__(None, None, None)

    def attach_client(self, client_id: str):
        if client_id in self._client_ids:
            return

        logfire.info(f"Attaching client: {client_id}")
        history_provider = lambda: NexusServerHistoryContext([])

        self._context_manager.add_context(
            f"history-{client_id}",
            history_provider,
            self._config.get("nexus.server.timeouts.history_context", 60)
        )

        self._client_ids.append(client_id)

    def detach_client(self, client_id: str):
        logfire.info(f"Detaching client: {client_id}")
        self._client_ids.remove(client_id)
        # TODO: implement remove context
        # self._context_manager.remove_context(f"history-{client_id}")

    def _check_client(self, client_id: str) -> None:
        if client_id not in self._client_ids:
            self.attach_client(client_id)
    
    async def prompt_agent(self, client_id: str, prompt: str) -> str:
        self._check_client(client_id)

        history_context_id = f"history-{client_id}"
        client_context_ids = ["agent-context", history_context_id]

        await self._context_manager.open(client_context_ids)

        logfire.info(f"Prompting agent: {prompt}")

        history_context = self._context_manager.get_context(history_context_id)
        print(history_context)
        assert isinstance(history_context, NexusServerHistoryContext)

        history = history_context.agent_history
        print(history)

        try:
            assert self._support_deps is not None, "Support dependencies not initialized"
            assert self._conversational_agent is not None, "Conversational agent not initialized"
            support_deps = self._support_deps
            result = await self._conversational_agent.run(
                prompt,
                message_history=history,
                deps=support_deps
            )
        except BaseException as e:
            logfire.exception("Exception in prompt_agent")
            raise e

        history_context.agent_history = result.all_messages()

        return result.output.text

    