from langchain_core.messages import HumanMessage
from accomodation_agent.agent import SYSTEM_MESSAGE, travel_info_agent
import ast
import re

def test_accommodation_retriever():
  question = "What is the highest rating of hotels in St Ives?"
  state = {"messages": [SYSTEM_MESSAGE, HumanMessage(content=question)]}
  state = travel_info_agent.invoke(state)
  reply = state["messages"][-1].content
  if isinstance(reply, list):
    reply = reply[0] if reply else ""
  if isinstance(reply, dict):
    reply = reply['text'] if 'text' in reply else str(reply)
  assert re.search(r"4[.,]99", reply)
