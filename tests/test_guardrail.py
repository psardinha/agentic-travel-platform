from orchestrator.guardrails.agent import guardrail_node
from langgraph.types import Command
from shared.AgentState import AgentState
from langchain_core.messages import HumanMessage

def test_guardrail_accept():
  state = {"messages": [HumanMessage(content="What is the weather like in Cornwall?")]}
  command = guardrail_node(state)
  assert command.goto == "router"

def test_guardrail_deny():
  state = {"messages": [HumanMessage(content="What was the football match score?")]}
  command = guardrail_node(state)
  assert command.goto == "__end__"  