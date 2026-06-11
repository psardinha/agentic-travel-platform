from typing import Literal, TypedDict


class AgentType(TypedDict):
    """Route a user question to relevant agent."""

    agent: Literal["travel_info_agent", "accommodation_booking_agent"]

if __name__ == "__main__":
    obj: AgentType = {"agent": "travel_info_agent"}
    print(obj)