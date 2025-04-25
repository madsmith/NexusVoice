import asyncio
from pydantic_ai import RunContext
from pydantic_ai.agent import AgentRunResult
import pytest
from nexusvoice.ai.pydantic_agent import (
    HomeAutomationAgent,
    HomeAutomationResponse,
    LocalClassifierAgent,
    FastClassifierAgent,
    NexusSupportDependencies,
    PydanticAgent,
    RequestType
)

from nexusvoice.core.config import load_config
from nexusvoice.utils.debug import TimeThis




# Fix for pydantic_ai using get_event_loop() in synchronous functions 
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
