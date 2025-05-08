from pydantic_ai.agent import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from nexusvoice.ai.types import ConversationResponse, NexusSupportDependencies

class ConversationalAgentFactory:
    @classmethod
    def create(cls, support_deps: NexusSupportDependencies) -> Agent[NexusSupportDependencies, ConversationResponse]:
        """
        Factory method to create a pydantic_ai.Agent for home automation intents.
        """
        config = support_deps.config
        provider = OpenAIProvider(
            api_key=config.get("agents.conversational.api_key", ""),
            base_url=config.get("agents.conversational.base_url", None)
        )
        model = OpenAIModel(
            model_name=config.get('agents.conversational.model', 'llama-3.2-3b-instruct'),
            provider=provider
        )
        system_prompt = config.get(
            "agents.conversational.system_prompt",
            "You are a helpful assistant. Provide clear and concise responses "
            "that are suitable for audio playback. Keep responses brief and natural."
        )
        return Agent[NexusSupportDependencies, ConversationResponse](
            model,
            system_prompt=system_prompt,
            retries=config.get("agents.conversational.retries", 1),
            deps_type=NexusSupportDependencies,
            result_type=ConversationResponse # type: ignore[arg-type]
        )