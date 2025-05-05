from typing import Optional
from pydantic_ai.agent import Agent
from pydantic_ai.providers import Provider
from pydantic_ai.providers.openai import AsyncOpenAI, OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from nexusvoice.ai.types import HomeAutomationResponseStruct, NexusSupportDependencies, HomeAutomationResponse

class HomeAutomationAgentFactory:
    @classmethod
    def create(cls, support_deps: NexusSupportDependencies, provider: Optional[Provider[AsyncOpenAI]] = None) -> Agent[NexusSupportDependencies, HomeAutomationResponse]:
        """
        Factory method to create a pydantic_ai.Agent for home automation intents.
        """
        config = support_deps.config
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
            result_type=HomeAutomationResponseStruct # type: ignore[arg-type]
        )