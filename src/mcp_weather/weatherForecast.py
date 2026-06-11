import random
from typing import Literal, get_args
from typing_extensions import TypedDict
from fastmcp import FastMCP
from mcp_weather import __version__
from shared.config import MCP_SERVER_PORT, MCP_SERVER_SERVLET_CTX_PATH

WeatherOption = Literal["sunny", "foggy", "rainy", "windy", "cloudy", "stormy"]

class WeatherForecast(TypedDict):
  town: str
  weather: WeatherOption
  temperature: int

mcp = FastMCP("mcp-weather-forecast", __version__)

@mcp.tool(description="Get weather conditions for a location.") 
async def get_weather_conditions(location: str) -> WeatherForecast:
  "Get weather conditions for a location."

  TEMP_MIN = 5
  TEMP_MAX = 40
  weather: WeatherOption = random.choice(get_args(WeatherOption))
  temperature: int = random.randint(TEMP_MIN, TEMP_MAX)
  print(f"Generated weather conditions for {location}: {weather}, {temperature}°C")
  return WeatherForecast(town=location, weather=weather, temperature=temperature)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=MCP_SERVER_PORT, path=MCP_SERVER_SERVLET_CTX_PATH)