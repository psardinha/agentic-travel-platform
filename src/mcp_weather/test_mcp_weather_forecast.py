from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import asyncio
import sys
from shared.config import MCP_SERVER_URL

transport = StreamableHttpTransport(url=MCP_SERVER_URL)
client = Client(transport)
async def main(town: str = "Lisbon"):
  # Connection is established here
  async with client:
    #print(f"Client connected: {client.is_connected()}")
    tools = await client.list_tools()
    #print(f"Available tools: {tools}")
    if any(tool.name == "get_weather_conditions" for tool in tools):
      result = await client.call_tool("get_weather_conditions", {"location": town})
      print(f"Call result: {result}")
      print(f"Weather forecast for {result.data.town}: {result.data.weather}, {result.data.temperature}°C")
    # Connection is closed automatically here
    #print(f"Client connected: {client.is_connected()}")

if __name__ == "__main__":
  if len(sys.argv) > 1:
    asyncio.run(main(sys.argv[1]))
  else:
    asyncio.run(main())
