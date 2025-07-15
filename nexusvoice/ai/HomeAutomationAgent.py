from typing import Optional
from pydantic_ai.agent import Agent
from pydantic_ai.providers import Provider
from pydantic_ai.providers.openai import AsyncOpenAI, OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from nexusvoice.ai.types import HomeAutomationResponseStruct, NexusSupportDependencies

class HomeAutomationAgentFactory:
    @classmethod
    def create(cls, support_deps: NexusSupportDependencies, provider: Optional[Provider[AsyncOpenAI]] = None) -> Agent[NexusSupportDependencies, HomeAutomationResponseStruct]:
        """
        Factory method to create a pydantic_ai.Agent for home automation intents.
        """
        config = support_deps.config

        mcp_server_names = config.get("agents.home_automation.mcp_servers", [])
        servers = []
        for server_name in mcp_server_names:
            if server_name not in support_deps.servers:
                raise ValueError(f"MCP server {server_name} not found")
            servers.append(support_deps.servers[server_name])
        
        provider = provider or OpenAIProvider(
            api_key=config.get("agents.home_automation.api_key", ""),
            base_url=config.get("agents.home_automation.base_url", None)
        )
        model = OpenAIModel(
            model_name=config.get('agents.home_automation.model', 'llama-3.2-3b-instruct'),
            provider=provider
        )
        system_prompt = config.get(
            "agent.home_automation.system_prompt",
            "You are a helpful assistant that controls a home automation system. "
            "You can control lights, fans, and shades in different rooms. "
            "When asked to perform a task, respond with a structured command. "
            "For informational queries, provide clear, concise responses suitable for audio playback."
        )
        return Agent[NexusSupportDependencies, HomeAutomationResponseStruct](
            model,
            system_prompt=system_prompt,
            retries=config.get("agents.home_automation.retries", 1),
            deps_type=NexusSupportDependencies,
            output_type=HomeAutomationResponseStruct,
            mcp_servers=servers
        )