from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar, Union
from typing_extensions import ParamSpec
from nexusvoice.ai.LocalClassifierAgent import LocalClassifierAgentFactory
from nexusvoice.core.config import NexusConfig
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers import Provider
from pydantic_ai.providers.openai import OpenAIProvider
from transformers import AutoConfig
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult, ToolFuncContext
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, TextPart, ModelResponsePart, ToolCallPart, UserPromptPart
import torch
from transformers import pipeline
from pathlib import Path

from nexusvoice.core.api.base import ModelResponse
from nexusvoice.utils.logging import get_logger
from nexusvoice.ai.types import NexusSupportDependencies, RequestType

logger = get_logger(__name__)

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT_DIR / "nexusvoice" / "models"


class HomeAutomationAction(BaseModel):
    intent: str = Field(..., description="The action to perform (e.g., turn_on, turn_off, raise, lower)")
    device: str = Field(..., description="The device to control (e.g., light, fan, shade)")
    room: str = Field(..., description="The room where the device is located")

class HomeAutomationResponseStruct(BaseModel):
    """Response from the home automation agent"""
    summary_message: str = Field(..., description="A short summary response indicating the success or failure status of the action completed.")

    @staticmethod
    def extract_message(response: "HomeAutomationResponse") -> str:
        if isinstance(response, str):
            return response
        return response.summary_message

HomeAutomationResponse = Union[HomeAutomationResponseStruct, str]

class ConversationResponse(BaseModel):
    """Response from the conversational agent"""
    text: str = Field(..., description="The response text")

BaseAgentRunResultType = TypeVar('BaseAgentRunResultType')
AgentDepsT = TypeVar('AgentDepsT')
ToolParamSpec = ParamSpec("ToolParamSpec")

class BaseAgent(ABC, Generic[BaseAgentRunResultType, ToolParamSpec]):
    def __init__(self, support_deps: NexusSupportDependencies):
        self._deps = support_deps
        self._config = support_deps.config

    @property
    def config(self) -> NexusConfig:
        return self._config

    @property
    def deps(self) -> NexusSupportDependencies:
        return self._deps
    
    @abstractmethod
    def run_sync(self, prompt: str) -> AgentRunResult[BaseAgentRunResultType]:
        pass

    @abstractmethod
    def register_tool(self, tool_fn: ToolFuncContext[NexusSupportDependencies, ToolParamSpec]):
        pass


class LocalClassifierAgent(BaseAgent[RequestType, ToolParamSpec]):
    """Local classifier using a fine-tuned DistilBERT model"""
    
    def __init__(self, support_deps: NexusSupportDependencies):
        super().__init__(support_deps)

        model_name = self.config.get("agents.classifier.model")
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            logger.warning(f"Fine-tuned model not found at {model_path}. Please run train_classifier.py first.")
            model = "distilbert-base-uncased"
        else:
            model = str(model_path)
        
        # Load the classifier
        self.classifier = pipeline(
            "text-classification",
            model=model,
            device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model_config = AutoConfig.from_pretrained(model_path)
        
        # Create the agent with our function model
        self._agent = Agent[NexusSupportDependencies, RequestType](
            FunctionModel(function=self._classifer, model_name="local-classifier"),
            system_prompt="",  # Not used for function model
            deps_type=NexusSupportDependencies,
            result_type=RequestType
        )
    
    def run_sync(self, prompt: str) -> AgentRunResult[RequestType]:
        return self._agent.run_sync(user_prompt=prompt, deps=self.deps)

    def _classifer(self, messages: List[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        # Select the proper tool, this expects a tool for RequestType
        result_tool = None
        for tool in agent_info.result_tools:
            if tool.parameters_json_schema['title'] == 'RequestType':
                result_tool = tool
                break

        if not result_tool:
            raise ValueError("No result tool found")

        # Get the last message which contains the text to classify
        text = None
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    text = str(part.content)
            
        if not text:
            raise ValueError("No message content found")
            
        outputs = self.classifier(text)

        # Pipeline returns a list of dicts with 'label' and 'score' keys
        result = outputs[0]  # type: ignore[index]
        label = str(result["label"])  # type: ignore[index]
        confidence = float(result["score"])  # type: ignore[index]

        label_id = self._model_config.label2id[label]

        # Our fine-tuned model uses 1 for home_automation, 0 for conversation
        type_str = "home_automation" if label_id == 1 else "conversation"
        request_type = RequestType(
            type=type_str,
            confidence=confidence
        )

        response_part = ToolCallPart(
            tool_name=result_tool.name,
            args=request_type.model_dump()
        )
        return ModelResponse(parts=[response_part])

    def register_tool(self, tool_fn: ToolFuncContext[NexusSupportDependencies, ToolParamSpec]):
        pass

class FastClassifierAgent(BaseAgent[RequestType, ToolParamSpec]):
    """Fast OpenAI classifier using a smaller, cheaper model"""
    
    def __init__(self, support_deps: NexusSupportDependencies):
        super().__init__(support_deps)

        provider = OpenAIProvider(api_key=self.config.get("openai.api_key", ""))
        model = OpenAIModel(
            model_name=self.config.get("agent.classifier.fast.model", "gpt-3.5-turbo"),
            provider=provider
        )

        self._agent = Agent[NexusSupportDependencies, RequestType](
            model,
            system_prompt=self._get_system_prompt(),
            deps_type=NexusSupportDependencies,
            result_type=RequestType,
            result_tool_name='final_result'
        )
    
    def run_sync(self, prompt: str) -> AgentRunResult[RequestType]:
        return self._agent.run_sync(user_prompt=prompt, deps=self.deps)

    def _get_system_prompt(self) -> str:
        default_prompt = (
            "You are a classifier that determines if a request is related to home automation "
            "or general conversation. Home automation requests involve controlling devices "
            "like lights, fans, or shades. Respond ONLY with the classification and confidence "
            "score as a JSON object with 'type' and 'confidence' fields. Type must be either "
            "'home_automation' or 'conversation'. Confidence must be between 0 and 1."
        )
        return self._config.get("agent.classifier.fast.system_prompt", default_prompt)

    def register_tool(self, tool_fn: ToolFuncContext[NexusSupportDependencies, ToolParamSpec]):
        self._agent.tool(tool_fn)

class HomeAutomationAgent(BaseAgent[HomeAutomationResponse, ToolParamSpec]):
    """Agent for home automation using pydantic_ai"""
    
    def __init__(self, support_deps: NexusSupportDependencies, provider: Optional[Provider[AsyncOpenAI]] = None):
        super().__init__(support_deps)

    
        provider = provider or OpenAIProvider(
            api_key=self.config.get("openai.api_key", ""),
            base_url=self.config.get("openai.base_url", None)
        )
        model = OpenAIModel(
            model_name=self.config.get('agents.home_automation.model', 'gpt-4-turbo-preview'),
            provider=provider
        )

        self._agent = Agent[NexusSupportDependencies, HomeAutomationResponse](
            model,
            system_prompt=self._get_system_prompt(),
            deps_type=NexusSupportDependencies,
            result_tool_name='report_final_state',
            result_type=HomeAutomationResponse # type: ignore[arg-type]
        )
    
    def run_sync(self, prompt: str) -> AgentRunResult[HomeAutomationResponse]:
        return self._agent.run_sync(user_prompt=prompt, deps=self.deps)

    def _get_system_prompt(self) -> str:
        default_prompt = (
            "You are a helpful assistant that controls a home automation system. "
            "You can control lights, fans, and shades in different rooms. "
            "When asked to perform a task, respond with a structured command. "
            "For informational queries, provide clear, concise responses suitable for audio playback."
        )
        return self._config.get("agent.home_automation.system_prompt", default_prompt)

    def register_tool(self, tool_fn: ToolFuncContext[NexusSupportDependencies, ToolParamSpec]):
        self._agent.tool(tool_fn)

class ConversationalAgent(BaseAgent[ConversationResponse, ToolParamSpec]):
    """Agent for general conversation using pydantic_ai"""
    
    def __init__(self, support_deps: NexusSupportDependencies):
        super().__init__(support_deps)

        provider = OpenAIProvider(api_key=self.config.get("openai.api_key", ""))
        model = OpenAIModel(
            model_name=self.config.get("agents.conversational.model", "gpt-4-turbo-preview"),
            provider=provider
        )

        self._agent = Agent[NexusSupportDependencies, ConversationResponse](
            model,
            system_prompt=self._get_system_prompt(),
            deps_type=NexusSupportDependencies,
            result_type=ConversationResponse
        )
    
    def run_sync(self, prompt: str) -> AgentRunResult[ConversationResponse]:
        return self._agent.run_sync(user_prompt=prompt, deps=self.deps)
    
    def _get_system_prompt(self) -> str:
        default_prompt = (
            "You are a helpful assistant. Provide clear and concise responses "
            "that are suitable for audio playback. Keep responses brief and natural."
        )
        return self._config.get("agent.conversational.system_prompt", default_prompt)

    def register_tool(self, tool_fn: ToolFuncContext[NexusSupportDependencies, ToolParamSpec]):
        self._agent.tool(tool_fn)

class PydanticAgentAPI:
    def __init__(self, config: NexusConfig, client_id: str):
        self.config = config
        self.client_id = client_id

        self._classifier_agent = None
        self._home_agent = None
        self._conversational_agent = None

    @property
    def classifier_agent(self) -> Agent[NexusSupportDependencies, RequestType]:
        assert self._classifier_agent is not None, "Classifier agent not initialized"
        return self._classifier_agent
    
    @property
    def home_agent(self) -> HomeAutomationAgent:
        assert self._home_agent is not None, "Home automation agent not initialized"
        return self._home_agent
    
    @property
    def conversational_agent(self) -> ConversationalAgent:
        assert self._conversational_agent is not None, "Conversational agent not initialized"
        return self._conversational_agent
    
    def initialize_agents(self):
        logger.debug("Initializing agents...")
        support_deps = NexusSupportDependencies(config=self.config)

        self._classifier_agent = LocalClassifierAgentFactory.create(support_deps)
        self._home_agent = HomeAutomationAgent(support_deps)
        self._conversational_agent = ConversationalAgent(support_deps)
    
    def initialize(self):
        self.initialize_agents()
    
    def start(self):
        logger.debug("Starting PydanticAgent...")
        self.initialize()
        pass

    def stop(self):
        logger.debug("Stopping PydanticAgent...")
        pass

    def process_request(self, text: str) -> ModelResponse:
        """Process a request and return a ModelResponse"""
        
        # First use the classifier to determine request type
        # Try local classifier first
        classification = self._classify_request(text)

        logger.debug(f"Request classified as {classification.type} with confidence {classification.confidence}")

        # If confident it's a home automation request, use the home automation agent
        if classification.type == "home_automation" and classification.confidence > 0.7:
            return self._process_home_automation(text)

        # Fall back to conversational response
        return self._process_conversational(text)

    def _classify_request(self, text: str) -> RequestType:
        try:
            deps = NexusSupportDependencies(config=self.config)
            result = self.classifier_agent.run_sync(text, deps=deps)
            return result.data
        except Exception as e:
            logger.warning(f"Local classifier failed: {e}, defaulting to conversation")
            # print stack trace
            import traceback
            traceback.print_exc()
            return RequestType(
                type="conversation",
                confidence=0.0
            )

    def _process_home_automation(self, text: str) -> ModelResponse:
        logger.debug("Processing home automation request...")
        try:
            result = self.home_agent.run_sync(text)
            if isinstance(result.data, HomeAutomationAction):
                tool_calls: List[ModelResponsePart] = [ToolCallPart(
                    tool_name="home_control",
                    args=result.data.model_dump()
                )]
                return ModelResponse(parts=tool_calls)
            elif isinstance(result.data, HomeAutomationResponse):
                message = result.data.summary_message if isinstance(result.data, HomeAutomationResponseStruct) else result.data
                return ModelResponse(parts=[TextPart(content=message)])
        except Exception as e:
            logger.debug(f"Home automation processing failed: {e}")
        return ModelResponse(parts=[])

    def _process_conversational(self, text: str) -> ModelResponse:
        logger.debug("Processing conversational request...")
        result = self.conversational_agent.run_sync(text)
        return ModelResponse(parts=[TextPart(content=result.data.text)])
