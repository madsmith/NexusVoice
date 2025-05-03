from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Generator, List
from nexusvoice.core.config import NexusConfig
import pydantic_core
from pydantic_ai.messages import ModelRequest, ModelResponsePart, RetryPromptPart, SystemPromptPart, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model, ModelMessage, ModelRequestParameters, ModelResponse, ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage
from pydantic_ai.agent import Agent
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, LlamaForCausalLM, PreTrainedTokenizerBase

from nexusvoice.ai.types import HomeAutomationResponse, NexusSupportDependencies
from nexusvoice.utils.logging import get_logger
import transformers
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = get_logger(__name__)

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT_DIR / "nexusvoice" / "models"

VALID_JSON_TYPE_MAPPING: dict[str, Any] = {
    'string': str,
    'integer': int,
    'number': float,
    'boolean': bool,
    'array': list,
    'object': dict,
    'null': type(None),
}

@dataclass(init=False)
class HomeAutomationModel(Model):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self._config = config

        self._model_name: str = self._config.get("agents.local_home_automation.model")

        self._device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self._model_name)
        self._tokenizer.add_special_tokens({"pad_token": "<|pad_id|>"})

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.bfloat16
        ).to(self._device)

        # Resize embeddings to match tokenizer with additional tokens
        self._model.resize_token_embeddings(len(self._tokenizer))

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelResponse, Usage]:
        # print()
        # self._print_messages(messages)
        # self._print_model_request_parameters(model_request_parameters)

        # Format the messages into a chat template
        chat_history = list(chain.from_iterable(self._map_message(m) for m in messages))

        # Inject tools into chat history
        # chat_history = self._inject_tools(chat_history, model_request_parameters)
        
        # print("Chat History:")
        # for message in chat_history:
        #     print(f"    {message}")

        all_tools = model_request_parameters.function_tools + model_request_parameters.result_tools
        tools = [self._tool_description(tool) for tool in all_tools]

        chat_input = self._tokenizer.apply_chat_template(
            chat_history,
            tools=tools, # type: ignore
            tokenize=False,
            add_generation_prompt=True,
            tools_in_user_message=False
        )
        assert isinstance(chat_input, str), "Chat input is not a string."
        # print("Chat Input:", chat_input)

        # Tokenize the input
        inputs = self._tokenizer.encode(
            chat_input,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True)
        inputs = inputs.to(self._device) # type: ignore
        input_length = inputs.shape[1]

        # Generate the output
        attention_mask = (inputs != self._tokenizer.pad_token_id).long().to(self._device)
        generation_params = {
            "attention_mask": attention_mask,
            "pad_token_id": self._tokenizer.pad_token_id,
            "max_length": 2048
        }

        with torch.no_grad():
            outputs = self._model.generate(inputs, **generation_params)
        outputs = outputs.to("cpu")

        # Sum up all input tokens for usage calculations
        input_tokens = sum(len(token) for token in inputs)
        output_tokens = sum(len(token) for token in outputs)

        # Output has chat template prefixed to it, remove inputs from output
        new_outputs = outputs[0, input_length:]

        decoded_output = self._tokenizer.decode(new_outputs, skip_special_tokens=True)

        # print("!!! Decoded Output:", decoded_output)

        # Check for tool calls
        tool_calls = self._try_extract_tool_calls(decoded_output, model_request_parameters)

        parts: list[ModelResponsePart] = []
        if tool_calls:
            parts = [ToolCallPart(tool_name=tool.tool_name, args=tool.args) for tool in tool_calls]
        else:
            parts = [TextPart(content=decoded_output)]
        # print("!!! Parts:", parts)
        
        model_response = ModelResponse(parts=parts)

        return model_response, Usage(1, input_tokens, output_tokens, input_tokens + output_tokens)

    @classmethod
    def _tool_description(cls, tool: ToolDefinition) -> dict[str, Any]:
        """
        Convert tool to a dict that can be used in the chat template 
        
        {
            'name': 'drink_beverage',
            'description': 'A function that drinks a beverage',
            'parameters': {
                'type': 'object',
                'properties': {
                    'beverage': {
                        'type': 'string',
                        'enum': ['tea', 'coffee'],
                        'description': 'The beverage to drink'
                        }
                    },
                'required': ['beverage']
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_json_schema
            }
        }
    
    @classmethod
    def _try_extract_tool_calls(cls, response: str, model_request_parameters: ModelRequestParameters) -> list[ToolCallPart]:
        """Scan the response for tool calls."""
        output_json: dict[str, Any] | None = pydantic_core.from_json(response, allow_partial='trailing-strings')

        if output_json:
            all_tools = model_request_parameters.function_tools + model_request_parameters.result_tools
            for tool in all_tools:
                if tool.name != output_json.get('name'):
                    # print("!!! Tool name does not match:", tool.name, output_json.get('name'))
                    continue
                params = output_json
                if 'parameters' in params:
                    params = params['parameters']
                is_valid = cls._validate_required_json_schema(params, tool.parameters_json_schema)
                if not is_valid:
                    # print("!!! Invalid JSON schema for tool:", tool.name)
                    # print("!!! JSON Schema:", tool.parameters_json_schema)
                    # print("!!! Output JSON:", output_json)
                    continue

                # The following part_id will be thrown away
                return [ToolCallPart(tool_name=tool.name, args=params)]
        return []
    
    @classmethod
    def _validate_required_json_schema(cls, json_dict: dict[str, Any], json_schema: dict[str, Any]) -> bool:
        """Validate that all required parameters in the JSON schema are present in the JSON dictionary."""
        required_params = json_schema.get('required', [])
        properties = json_schema.get('properties', {})

        for param in required_params:
            if param not in json_dict:
                # print("!!! Required parameter not found:", param)
                return False

            param_schema = properties.get(param, {})
            param_type = param_schema.get('type')
            param_items_type = param_schema.get('items', {}).get('type')

            if param_type == 'array' and param_items_type:
                if not isinstance(json_dict[param], list):
                    # print("!!! Required parameter is not a list:", param)
                    return False
                for item in json_dict[param]:
                    if not isinstance(item, VALID_JSON_TYPE_MAPPING[param_items_type]):
                        # print("!!! Required parameter is not a valid type:", param)
                        return False
            elif param_type and not isinstance(json_dict[param], VALID_JSON_TYPE_MAPPING[param_type]):
                # print("!!! Required parameter is not a valid type:", param)
                return False

            if isinstance(json_dict[param], dict) and 'properties' in param_schema:
                nested_schema = param_schema
                if not cls._validate_required_json_schema(json_dict[param], nested_schema):
                    # print("!!! Required parameter is invalid schema:", param)
                    return False

        return True
    
    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Generator[dict[str, str], None, None]:
        """Yield dicts for SystemPromptPart and UserPromptPart, error for others."""
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield {"role": "system", "content": part.content}
            elif isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    yield {"role": "user", "content": part.content}
                else:
                    for content in part.content:
                        if isinstance(content, str):
                            yield {"role": "user", "content": content}
                        else:
                            logger.warning(f"Non-string content in user prompt part, skipping: {str(content)}.")
            elif isinstance(part, ToolReturnPart):
                yield {
                    "role": "tool",
                    "name": part.tool_name,
                    "tool_call_id": part.tool_call_id,
                    "content": part.model_response_str()  # must be a string
                }
            elif isinstance(part, RetryPromptPart):
                msg_items = [("role", "tool")]

                if part.tool_name:
                    msg_items.append(("name", part.tool_name))

                if part.tool_call_id:
                    msg_items.append(("tool_call_id", part.tool_call_id))

                msg_items.append(("content", part.model_response()))

                yield dict(msg_items)
            else:
                raise ValueError(f"Unsupported message part type: {type(part)}")

    @classmethod
    def _map_assistant_message(cls, message: ModelResponse) -> Generator[dict[str, str], None, None]:
        for part in message.parts:
            if isinstance(part, TextPart):
                yield {"role": "assistant", "content": part.content}
            elif isinstance(part, ToolCallPart):
                if isinstance(part.args, str):
                    args = part.args
                else:
                    if 'parameters' in part.args:
                        args = part.args['parameters']
                    else:
                        args = part.args

                tool_content = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": part.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": args
                            }
                        }
                    ]
                }
                yield tool_content
            else:
                raise ValueError(f"Unsupported message part type: {type(part)}")

    @classmethod
    def _map_message(cls, message: ModelMessage) -> Generator[dict[str, str], None, None]:
        if isinstance(message, ModelRequest):
            yield from cls._map_user_message(message)
        elif isinstance(message, ModelResponse):
            yield from cls._map_assistant_message(message)
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    def _inject_tools(self, chat_history: list[dict[str, str]], model_request_parameters: ModelRequestParameters) -> list[dict[str, str]]:
        modified_chat_history = chat_history.copy()
        
        # Locate system prompt
        system_prompt_index = None
        for message in modified_chat_history:
            if message['role'] == 'system':
                system_prompt_index = modified_chat_history.index(message)
                break
        
        if system_prompt_index is None:
            raise ValueError("System prompt not found in chat history")
        
        # Inject tools
        tools_prompt = "\n\nAvailable tools:\n"
        for tool in model_request_parameters.function_tools:
            tools_prompt += f"{tool.name}: {tool.description}\n"
        modified_chat_history[system_prompt_index]['content'] += "\n\n" + tools_prompt
        return modified_chat_history

    def _get_user_prompt_text(self, messages: list[ModelMessage]) -> str:
        last_message = messages[-1]
        assert isinstance(last_message, ModelRequest), "Expected a Model Request as last message."
        content_parts = [
            part.content
            for part in last_message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ]
        text = "".join(content_parts)
        return text

    def _get_result_tool(self, model_request_parameters: ModelRequestParameters) -> ToolDefinition:
        for tool in model_request_parameters.result_tools:
            if tool.parameters_json_schema['title'] == 'HomeAutomationRequestType':
                return tool
        raise ValueError("No result tool found")

    @classmethod
    def _print_messages(cls, messages: list[ModelMessage]):
        for message in messages:
            print(f" + {message.__class__.__name__}")
            for i, part in enumerate(message.parts):
                last_part = i == len(message.parts) - 1
                if isinstance(part, TextPart):
                    content = part.content
                elif isinstance(part, ToolCallPart):
                    content = f"Call: {part.tool_name} {part.args}"
                elif isinstance(part, ToolReturnPart):
                    content = f"Return: {part.tool_name} {part.model_response_str()}"
                elif isinstance(part, UserPromptPart):
                    content = part.content
                elif isinstance(part, SystemPromptPart):
                    content = part.content
                elif isinstance(part, RetryPromptPart):
                    content = part.content
                else:
                    raise ValueError(f"Unsupported message part type: {type(part)}")
                prefix_char = "├" if not last_part else "└"
                print(f"  {prefix_char} {part.__class__.__name__}: {content}")

    @classmethod
    def _print_model_request_parameters(cls, model_request_parameters: ModelRequestParameters):
        print("Model Request Parameters:")
        for func in model_request_parameters.function_tools:
            print(f"    Tool: {func.name}: {func.description}")
        for func in model_request_parameters.result_tools:
            print(f"    Result: {func.name}: {func.description}")
    
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "local"

class LocalHomeAutomationAgentFactory():
    @classmethod
    def create(cls, support_deps: NexusSupportDependencies):
        model = HomeAutomationModel(support_deps.config)
        agent = Agent[NexusSupportDependencies, HomeAutomationResponse](
            model=model,
            system_prompt=cls._get_system_prompt(support_deps.config),
            deps_type=NexusSupportDependencies,
            result_type=HomeAutomationResponse # type: ignore[arg-type]
        )
        return agent

    @classmethod
    def _get_system_prompt(cls, config: NexusConfig) -> str:
        default_prompt = (
            "You are a helpful assistant that controls a home automation system. "
            "You can control lights, fans, and shades in different rooms. "
            "When asked to perform a task, respond with a structured command. "
            "For informational queries, provide clear, concise responses suitable for audio playback."
        )
        return config.get("agents.local_home_automation.system_prompt", default_prompt)
