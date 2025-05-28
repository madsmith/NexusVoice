from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_core import to_jsonable_python
import json

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter  

from nexusvoice.core.config import load_config

config = load_config()

api_key = config.get("openai.api_key")
provider = OpenAIProvider(api_key=api_key)
model = OpenAIModel(model_name="gpt-4o", provider=provider)

agent = Agent(model, system_prompt='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
history_step_1 = result1.all_messages()
print("All Messages")
for message in history_step_1:
    print(message)

print("As JSON")
as_python_objects = to_jsonable_python(history_step_1)  
print(json.dumps(as_python_objects, indent=2))
same_history_as_step_1 = ModelMessagesTypeAdapter.validate_python(as_python_objects)

result2 = agent.run_sync(  
    'What makes that joke funny?.', message_history=same_history_as_step_1
)
print(result2.data)
