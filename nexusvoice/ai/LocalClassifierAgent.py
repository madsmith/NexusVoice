
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
from nexusvoice.core.config import NexusConfig
from pydantic_ai.messages import ModelRequest, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import Model, ModelMessage, ModelRequestParameters, ModelResponse, ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage
from pydantic_ai.agent import Agent

from nexusvoice.ai.types import RequestType, NexusSupportDependencies
from nexusvoice.utils.logging import get_logger
import torch
from transformers import AutoConfig, pipeline
from transformers.pipelines import PreTrainedTokenizer

logger = get_logger(__name__)

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT_DIR / "nexusvoice" / "models"

@dataclass(init=False)
class IntentClassifierModel(Model):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self._config = config

        self._model_name = self._config.get("agents.classifier.model")
        model_path = MODEL_DIR / self._model_name

        if not model_path.exists():
            logger.warning(f"Fine-tuned model not found at {model_path}. Please run train_classifier.py first.")
            model = "distilbert-base-uncased"
        else:
            model = str(model_path)

        # Load torch device
        device = torch.device(
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )

        # Load the classifier
        self._classifier = pipeline(
            "text-classification",
            model=model,
            device=device
        )

        self._model_config = AutoConfig.from_pretrained(model_path)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelResponse, Usage]:

        text = self._get_user_prompt_text(messages)

        outputs = self._classifier(text)

        assert isinstance(outputs, list), "Classifier did not return list of outputs"
        assert len(outputs) == 1, "Classifier did not return single result"
        assert isinstance(outputs[0], dict), "Classifier result is not a dict"
        
        label = str(outputs[0]["label"])
        confidence = float(outputs[0]["score"])
        label_id = self._model_config.label2id[label]

        request_type = RequestType(
            type="home_automation" if label_id == 1 else "conversation",
            confidence=confidence
        )

        # Build response
        result_tool = self._get_result_tool(model_request_parameters)

        if result_tool is not None:
            part = ToolCallPart(
                tool_name=result_tool.name,
                args=request_type.model_dump()
            )
        else:
            part = TextPart(content=request_type.model_dump_json())
        
        # Tally usage for the model
        tokenizer: PreTrainedTokenizer | None= self._classifier.tokenizer
        usage = Usage(
            requests=1,
            request_tokens=len(tokenizer(text).input_ids) if tokenizer is not None else 1,
            response_tokens=len(self._model_config.label2id)
        )
        
        return ModelResponse(parts=[part]), usage
    
    def _get_user_prompt_text(self, messages: list[ModelMessage]) -> str:
        last_message = messages[-1]
        assert isinstance(last_message, ModelRequest), "Expected a Model Request as last message."

        # Get the content from the last message
        content_parts = [
            part.content 
            for part in last_message.parts 
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ]

        text = "".join(content_parts)
        return text
    
    def _get_result_tool(self, model_request_parameters: ModelRequestParameters) -> ToolDefinition:
        for tool in model_request_parameters.result_tools:
            if tool.parameters_json_schema['title'] == 'RequestType':
                return tool
        raise ValueError("No result tool found")

    @property
    def model_name(self) -> str:
        return self._config.get("agents.classifier.model")

    @property
    def system(self) -> str:
        return "local"
        

class LocalClassifierAgentFactory():
    @classmethod
    def create(cls, support_deps: NexusSupportDependencies):
        model = IntentClassifierModel(support_deps.config)

        agent = Agent[NexusSupportDependencies, RequestType](
            model=model,
            deps_type=NexusSupportDependencies,
            result_type=RequestType
        )

        return agent