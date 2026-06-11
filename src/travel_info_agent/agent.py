from fastapi import FastAPI
import uvicorn
import sys

from docsDB.store import get_collection
from docsDB.questionDemux.make_queries import MultiQueryChainFactory
from docsDB.vectorialRetriever.retriever import getDefaultRetriever
from docsDB.questionDemux.rrf import reciprocal_rank_fusion
from core.llm_provider import LLMProvider
from shared.AgentState import AgentState
from travel_info_agent.ChatRequest import ChatRequest
from travel_info_agent.ChatResponse import ChatResponse
from shared.config import TRAVEL_INFO_AGENT_A2A_PORT, MCP_SERVER_URL, MAX_DOCS_PASSED_TO_LLM, TRAVEL_INFO_AGENT_SERVLET_CTX_PATH

import asyncio
from typing import Dict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END

from langgraph.prebuilt import tools_condition
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

def docs_retriver(llm, collection):
  multi_query_chain = MultiQueryChainFactory.get_instance(llm)
  retriever = getDefaultRetriever(collection)
  return multi_query_chain | retriever.map() | reciprocal_rank_fusion | \
         RunnableLambda(lambda x: x[0:MAX_DOCS_PASSED_TO_LLM])

docsCollection = get_collection()
from shared.config import OPENAI_API_KEY
llm = LLMProvider.get_llm(model="gpt-4o-mini", api_key=OPENAI_API_KEY, use_responses_api=True)
rag_fusion_retrieval_chain = docs_retriver(llm, docsCollection)
app = FastAPI()

@tool(description="Search travel information about destinations in England.")
def search_travel_info(query: str) -> str:
  """Search embedded WikiVoyage content for information about destinations in England."""
  print("\n\nSearch travel information tool about destinations in England tool being called\n\n")
  docs = rag_fusion_retrieval_chain.invoke(query)
  top = docs[:4] if isinstance(docs, list) else docs
  data_retrieved = "\n---\n".join(d[0].page_content for d in top)
  #print(f"\n\n~~~~~~~~~~\nData retrived: {data_retrieved}\n~~~~~~~~~~\n\n")
  return data_retrieved

@tool(description="Get the weather forecast, given a town name.")
def weather_forecast(town: str) -> dict:
  """Get a mock weather forecast for a given town. Returns a WeatherForecast object with weather and temperature."""
  print("\n\nForecast tool being called\n\n")

  async def _call():
    transport = StreamableHttpTransport(url=MCP_SERVER_URL)
    async with Client(transport) as client:
      return await client.call_tool("get_weather_conditions", {"location": town})
    
  try:
    forecast = asyncio.run(_call())
  except Exception as e:
    return {"error": f"Weather request failed for '{town}': {str(e)}"}
  if not forecast:
    return {"error": f"No weather data available for '{town}'."}
  try:
    data = forecast.data
    return {"town": getattr(data, "town", town),
            "weather": getattr(data, "weather", "unknown"),
            "temperature": getattr(data, "temperature", None)}
  except Exception as e:
    return {"error": f"Invalid forecast format for '{town}': {str(e)}"}


TOOLS = [search_travel_info, weather_forecast] 
llm_with_tools = llm.bind_tools(TOOLS) 

def llm_node(state: AgentState):
  """LLM node that decides whether to call the search tool."""
  current_messages = state["messages"]
  print(f"\n++++++++++++ LLM node interactions++++\n")
  for ndx, msg in enumerate(current_messages, start=1):
    print(f"Msg # {ndx}: {msg}\n")
  print(f"\n++++++++++++\n")
  response_message = llm_with_tools.invoke(current_messages)
  return {"messages": [response_message]}


tools_execution_node = ToolNode(TOOLS)

builder = StateGraph(AgentState) 
builder.add_node("llm_node", llm_node) 
builder.add_node("tools", tools_execution_node) 
builder.add_conditional_edges("llm_node", tools_condition) 
builder.add_edge("tools", "llm_node") 
builder.set_entry_point("llm_node") 
travel_info_agent = builder.compile() 
sessions: Dict[str, AgentState] = {}
SYSTEM_MESSAGE = SystemMessage(content=("You are a helpful assistant "
                                        "that can search travel information and get the weather forecast.\n" 
                                        "Only use the tools to find the information you need (including town names).")) 

@app.post(TRAVEL_INFO_AGENT_SERVLET_CTX_PATH)
def chat(req: ChatRequest) -> ChatResponse:
  session_id = req.session_id
  if session_id not in sessions:
    sessions[session_id] = {"messages": [SYSTEM_MESSAGE]}
  state = sessions[session_id]
  print("Travel Info agent received a request in A2A interface\n - Session ID: {session_id}\n - Question  : {req.message}")
  state["messages"].append(HumanMessage(content=req.message))
  result = travel_info_agent.invoke(state)
  sessions[session_id] = result  # persist updated state
  if isinstance(result["messages"][-1].content, list):
    return ChatResponse(session_id=session_id, response=result["messages"][-1].content[0]["text"])
  else:
    return ChatResponse(session_id=session_id, response=result["messages"][-1].content)


def chat_loop():
  print("UK Travel Assistant (type 'exit' to quit)")
  state = {"messages": [SYSTEM_MESSAGE]}
  while True:
    user_input = input("You: ").strip() 
    if user_input.lower() in {"exit", "quit"}: 
      break
    state["messages"].append(HumanMessage(content=user_input))
    state = travel_info_agent.invoke(state) 
    response_msg = state["messages"][-1]
    print(f" -->{response_msg.content[0]['text']}<--")
    print(f"Assistant: {response_msg.content}\n")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1].lower() == "a2a":
    uvicorn.run("travel_info_agent.agent:app", host="0.0.0.0", port=TRAVEL_INFO_AGENT_A2A_PORT)
  else:
    chat_loop()
  print("The end")