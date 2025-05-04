from typing import Optional
from pydantic_ai.agent import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from nexusvoice.ai.types import NexusSupportDependencies, HomeAutomationResponse

class HomeAutomationAgentFactory:
    @classmethod
    def create(cls, support_deps: NexusSupportDependencies) -> Agent[NexusSupportDependencies, HomeAutomationResponse]:
        """
        Factory method to create a pydantic_ai.Agent for home automation intents.
        """
        config = support_deps.config
        provider = OpenAIProvider(
            api_key=config.get("openai.api_key", ""),
            base_url=config.get("openai.base_url", None)
        )
        model = OpenAIModel(
            model_name=config.get('agents.home_automation.model', 'gpt-4-turbo-preview'),
            provider=provider
        )
        system_prompt = config.get(
            "agent.home_automation.system_prompt",
            "You are a helpful assistant that controls a home automation system. "
            "You can control lights, fans, and shades in different rooms. "
            "When asked to perform a task, respond with a structured command. "
            "For informational queries, provide clear, concise responses suitable for audio playback."
        )
        return Agent[NexusSupportDependencies, HomeAutomationResponse](
            model,
            system_prompt=system_prompt,
            deps_type=NexusSupportDependencies,
            result_type=HomeAutomationResponse # type: ignore[arg-type]
        )