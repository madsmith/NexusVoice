import os
import requests

def get_weather_tool(input: dict):
    city = input.get("city", "New York")
    key = os.getenv("WEATHER_API_KEY")

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