from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import asyncio
import sys
from shared.config import MCP_SERVER_URL

EXC_CONN_FAILURE_MSG = "Client failed to connect: All connection attempts failed"

def test_weather_forecast():
  town = "Faro"

  async def _call():
    transport = StreamableHttpTransport(url=MCP_SERVER_URL)
    async with Client(transport) as client:
      return await client.call_tool("get_weather_conditions", {"location": town})

  try:
    forecast = asyncio.run(_call())
  except Exception as e:
    if isinstance(e, RuntimeError) and str(e) == EXC_CONN_FAILURE_MSG:
      print(("\n================================================================\n"
             "Make sure the MCP server for weather forecast service is running"
             "\n================================================================\n"))
    assert False
    return

  assert getattr(forecast.data, "town", "??") == town
  assert type(getattr(forecast.data, "temperature", "??")) is int
  assert type(getattr(forecast.data, "weather", 0)) is str