from core.llm_provider import LLMProvider

from typing import Dict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition

from shared.AgentState import AgentState

from shared.config import OPENAI_API_KEY, DB_URI

llm = LLMProvider.get_llm(model="gpt-4o-mini", api_key=OPENAI_API_KEY, use_responses_api=True)
hotel_db = SQLDatabase.from_uri(DB_URI)
hotel_db_toolkit = SQLDatabaseToolkit(db=hotel_db, llm=llm)
TOOLS = hotel_db_toolkit.get_tools()
llm_with_tools = llm.bind_tools(TOOLS) 

def llm_node(state: AgentState):
  """LLM node that decides whether to call the search tool."""
  current_messages = state["messages"]
  #print(f"\n++++++++++++ Accomodation LLM node interactions++++\n")
  #for ndx, msg in enumerate(current_messages, start=1):
  #  print(f"Msg # {ndx}: {msg}\n")
  #print(f"\n++++++++++++\n")
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

SYSTEM_MESSAGE = SystemMessage(content=("You are a hotel booking assistant.\n"
                                        "Read database schema via tools if needed.\n"
                                        "The hotels table defines the hotels/BB in each town, with name, rating, location...\n"
                                        "hotel_room_offers table defines per hotel how many double and single rooms are " 
                                        "available and the prices per day. "
                                        "When asked about availability:\n"
                                        " - Ask if it is not specified the number of single rooms and double rooms requested,\n"
                                        " - Check the database,\n"
                                        " - Use SQL tools,\n"
                                        " - Never invent data,\n"
                                        " - Return hotel names, rating and prices per individual type of room and the total price for the request.")) 

def agent_launch(state: AgentState, config: RunnableConfig) -> dict:
  state["messages"] += [SYSTEM_MESSAGE]
  state = travel_info_agent.invoke(state)
  print(f"Accomodation assistant: {state['messages'][-1].content}\n")
  return {"messages": [AIMessage(content=state["messages"][-1].content)]}  

def chat_loop():
  print("UK Accomodation Assistant (type 'exit' to quit)")
  state = {"messages": [SYSTEM_MESSAGE]}
  while True:
    user_input = input("You: ").strip() 
    if user_input.lower() in {"exit", "quit"}: 
      break
    state["messages"].append(HumanMessage(content=user_input))
    state = travel_info_agent.invoke(state) 
    response_msg = state["messages"][-1] 
    print(f"Assistant: {response_msg.content}\n")


if __name__ == "__main__":
  chat_loop()
  print("The end")