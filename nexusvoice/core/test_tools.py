import requests
import datetime

from nexusvoice.utils.logging import get_logger

logger = get_logger(__name__)

from nexusvoice.core.api.tool_registry import register_tool
from nexusvoice.core.config import load_config

@register_tool("get_weather")
def get_weather_tool(input: dict):
    config = load_config()

    city = input.get("city", "New York")
    key = config.get("tools.weather.api_key", None)

    if not key:
        raise ValueError("API key for weather tool not found in config.")

    resp = requests.get("https://api.weatherapi.com/v1/current.json", params={
        "key": key,
        "q": city
    })
    data = resp.json()

    return {
        "city": data["location"]["name"],
        "temp_f": data["current"]["temp_f"],
        "condition": data["current"]["condition"]["text"]
    }

@register_tool("get_date_and_time")
def get_date_and_time(input: dict):
    now = datetime.datetime.now()
    return {
        "time": now.strftime("%Y-%m-%d %H:%M:%S")
    }




