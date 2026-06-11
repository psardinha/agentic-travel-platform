from typing import TypedDict


class GuardrailDecision(TypedDict):
  """Stores the result of a guardrail decision"""

  is_travel: bool
  reason: str