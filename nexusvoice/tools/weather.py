import os
from typing import Optional
from nexusvoice.ai.pydantic_agent import NexusSupportDependencies
from nexusvoice.core.config import load_config
from pydantic_ai.agent import RunContext
import requests

# TODO: put config in RunContext
def get_weather_tool(ctx: RunContext[NexusSupportDependencies], city: str, state: Optional[str] = None) -> dict:
    """
    Get current weather information for a specified city.
    
    Args:
        ctx: The run context
        city: The name of the city to check
    """
    config = ctx.deps.config

    key = config.get("tools.weather.api_key", None)

    if not key:
        raise ValueError("API key for weather tool not found in config.")

    query = city if state is None else f"{city}, {state}"
    resp = requests.get("https://api.weatherapi.com/v1/current.json", params={
        "key": key,
        "q": query
    })
    data = resp.json()

    return {
        "city": data["location"]["name"],
        "temp_f": data["current"]["temp_f"],
        "condition": data["current"]["condition"]["text"]
    }