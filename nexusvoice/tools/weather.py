from typing import Optional
from nexusvoice.ai.types import NexusSupportDependencies
from pydantic_ai import RunContext
import requests

# TODO: put config in RunContext
def get_weather(ctx: RunContext[NexusSupportDependencies], city: str, state: Optional[str] = None) -> dict:
    """
    Get current weather information for a specified city.
    
    Args:
        ctx (RunContext[NexusSupportDependencies]): The run context.
        city (str): The city to get the weather for.
        state (str, optional): The state to get the weather for.

    Returns:
        dict: The current weather, with the following keys:
            - city: The city name
            - temp_f: The current temperature in Fahrenheit
            - condition: The current weather condition
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