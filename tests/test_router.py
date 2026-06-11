from orchestrator.agent import router_agent_node
from langgraph.types import Command
from shared.AgentState import AgentState
from langchain_core.messages import HumanMessage


import warnings
warnings.filterwarnings("error", category=UserWarning)


def test_router_travel_info():
  state = {"messages": [HumanMessage(content="What is the weather like in Cornwall?")]}
  command = router_agent_node(state)
  assert command.goto == "travel_info_agent"

# def test_router_accomodation():
#   state = {"messages": [HumanMessage(content="Are there rooms available in Newquay?")]}
#   command = router_agent_node(state)
#   assert command.goto == "accommodation_booking_agent"  