import requests
from core.llm_provider import LLMProvider

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from shared.AgentState import AgentState
from orchestrator.AgentType import AgentType
from orchestrator.guardrails.agent import guardrail_node
from accomodation_agent.agent import agent_launch
from shared.config import OPENAI_API_KEY, TRAVEL_INFO_AGENT_URL

llm = LLMProvider.get_llm(api_key=OPENAI_API_KEY)
llm_router = llm.with_structured_output(AgentType)

ROUTER_SYSTEM_PROMPT = ("You are a router. Given the following user message, "
                        "decide if it is a travel information question "
                        "(about destinations, attractions, or general travel info) "
                        "or an accommodation booking question (about hotels, "
                        "room availability, or prices).\n"
                        "If it is a travel information question, "
                        "make agent property value 'travel_info_agent'.\n"
                        "If it is an accommodation booking question, "
                        "make agent property value 'accommodation_booking_agent', instead.")

def router_agent_node(state: AgentState) -> Command:
  """Router node: decides which agent should handle the user query."""
  messages = state["messages"] 
  last_msg = messages[-1] if messages else None 
  if isinstance(last_msg, HumanMessage): 
    user_input = last_msg.content 
  else:
    user_input = ""
  #print(f"User input for router_agent_node: {user_input}") 
  router_messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                     HumanMessage(content=user_input)]
  router_response = llm_router.invoke(router_messages)
  if not isinstance(router_response, dict) or "agent" not in router_response:
    return Command(update={"messages": [AIMessage("Sorry, I am missing your purpose. Please rephrase your question")]}, goto=END)
  print(f"Router reply: {router_response['agent']}")
  return Command(goto=router_response['agent'])

def travel_info_agent(state: AgentState, config: RunnableConfig) -> dict:
  user_message = state["messages"][-1].content
  session_id = config["configurable"]["thread_id"]
  payload = {"session_id": session_id, "message": user_message}
  try: 
    response = requests.post(TRAVEL_INFO_AGENT_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return {"messages": [AIMessage(content=data.get("reply", str(data["response"])))]}
  except Exception as exc:
    return {"messages": [AIMessage(content="Error contacting Travel Info agent: " + str(exc))]}
  #return {"messages": [AIMessage(content=f"Travel Info Agent\nThreadID: {config['configurable']['thread_id']}\nYou asked: {state['messages'][-1].content}")]}


def accommodation_booking_agent(state: AgentState, config: RunnableConfig) -> dict:
  return agent_launch(state, config)

# def accommodation_booking_agent(state: AgentState) -> dict:
#   return {"messages": [AIMessage(content=f"Accomodation Agent\nYou asked: {state['messages'][-1]}")]}

# builder = StateGraph(AgentState) 
# builder.add_node("router_agent", router_agent_node) 
# builder.add_node("travel_info_agent", travel_info_agent) 
# builder.add_node("accommodation_booking_agent", accommodation_booking_agent) 
# builder.add_edge("travel_info_agent", END) 
# builder.add_edge("accommodation_booking_agent", END)
# builder.set_entry_point("router_agent") 
# travel_orchestrator = builder.compile() 

checkpointer = MemorySaver()
builder = StateGraph(AgentState) 
builder.add_node("guardrail", guardrail_node)
builder.add_node("router", router_agent_node)
builder.add_node("accommodation_booking_agent", accommodation_booking_agent) 
builder.add_node("travel_info_agent", travel_info_agent)
# Next transitions are resolved via direct jumps via Command
#   guardrail --> router
#   guardrail --> END
#   router    --> travel_info_agent
#   router    --> accommodation_booking_agent
builder.add_edge("travel_info_agent", END)
builder.add_edge("accommodation_booking_agent", END)
builder.set_entry_point("guardrail") 
travel_orchestrator = builder.compile(checkpointer=checkpointer)

def chat_loop():
  print("UK Touristic Cornwall Assistant (type 'exit' to quit)")
  config = {"configurable": {"thread_id": "user-123"}}
  state = {"messages": []}
  while True:
    user_input = input("You: ").strip() 
    if user_input.lower() in {"exit", "quit"}:
      break
    state["messages"].append(HumanMessage(content=user_input))
    state = travel_orchestrator.invoke(state, config=config) 
    response_msg = state["messages"][-1].content
    if isinstance(response_msg, list):
      response_msg = response_msg[0] if response_msg else ""
    if isinstance(response_msg, dict):
      response_msg = response_msg['text'] if 'text' in response_msg else str(response_msg)
    else:  
      response_msg =  response_msg if isinstance(response_msg, str) else str(response_msg) 
    print(f"Assistant: {response_msg}\n")


if __name__ == "__main__":
  chat_loop()
  print("The end")