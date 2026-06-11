from core.llm_provider import LLMProvider

from typing import Dict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.types import Command

from shared.AgentState import AgentState
from orchestrator.guardrails.GuardrailDecision import GuardrailDecision

from shared.config import OPENAI_API_KEY
llm = LLMProvider.get_llm(api_key=OPENAI_API_KEY)
llm_guardrail = llm.with_structured_output(GuardrailDecision)

GUARDRAIL_SYSTEM_PROMPT = ("You are a strict classifier. Given the user's last message, respond with whether "
                           "it is travel-related. Travel-related queries include destinations, attractions, "
                           "lodging (hotels), room availability, prices, or weather in Cornwall/England.")

REFUSAL_INSTRUCTION = ("Sorry, I can only help with travel-related " 
                       "questions (destinations, attractions, lodging, "
                       "prices, availability, or weather in Cornwall/England). "
                       "Please rephrase your request to be travel-related.")

def guardrail_node(state: AgentState) -> Command:
  """Guardrail node: decides whether the question is related to touristic 
     information, weather or accomodations in Cornwall/England"""
  messages = state["messages"] 
  last_msg = messages[-1] if messages else None 
  if isinstance(last_msg, HumanMessage): 
    user_input = last_msg.content 
  else:
    user_input = ""
  msgToAnalyze = [SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
                  HumanMessage(content=user_input)]
  guardrail_decision = llm_guardrail.invoke(msgToAnalyze)
  #print(f"Guardrail reply: {guardrail_decision}")
  if not guardrail_decision["is_travel"]:
    # Return refusal directly as an AI message and shortcut to END via a dedicated node
    return Command(update={"messages": [AIMessage(content=guardrail_decision["reason"] + "\n" + REFUSAL_INSTRUCTION)]}, goto=END)
  return Command(update=state, goto="router")



def router_node(state: AgentState) -> dict:
  return {"messages": [AIMessage(content=f"Router node\nYou asked: {state['messages'][-1]}")]}

builder = StateGraph(AgentState) 
builder.add_node("guardrail", guardrail_node) 
builder.add_node("router", router_node) 
builder.add_edge("guardrail", END)
builder.add_edge("router", END) 
builder.set_entry_point("guardrail") 
guardrail_checker = builder.compile() 


def chat_loop():
  print("Guardrail Assistant (type 'exit' to quit)")
  state = {"messages": []}
  while True:
    user_input = input("You: ").strip() 
    if user_input.lower() in {"exit", "quit"}: 
      break
    state["messages"].append(HumanMessage(content=user_input))
    state = guardrail_checker.invoke(state) 
    response_msg = state["messages"][-1] 
    print(f"Assistant: {response_msg}\n")


if __name__ == "__main__":
  chat_loop()
  print("The end")