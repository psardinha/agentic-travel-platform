from pydantic import BaseModel, Field

class DestinationSearch(BaseModel):
  """Search over a vector database of tourist destinations."""
  
  content_search: str = Field("", description="Similarity search query applied to tourist destinations.")
  
  destination: str = Field(..., description="The specific UK destination to be searched.")

  region: str = Field(..., description="The name of the UK region to be searched.")

  def pretty_print(self) -> None:
    for field_name, field_info in self.__class__.model_fields.items():
      value = getattr(self, field_name)
      default = field_info.default
      if value is not None and value != default:
        print(f"{field_name}: {value}")
