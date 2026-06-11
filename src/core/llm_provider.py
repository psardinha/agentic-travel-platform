from langchain_openai import ChatOpenAI

class LLMProvider:
   _instances = None

   @classmethod
   def get_llm(cls, model="gpt-5-nano", temperature=0.0, api_key=None, **kwargs):
     if cls._instances is None:
        cls._instances = {}
     kwargs_key = frozenset(sorted(kwargs.items()))
     key = (model, temperature, api_key, kwargs_key)
     if key in cls._instances:
       return cls._instances[key]
     new_instance = ChatOpenAI(model=model, temperature=temperature, api_key=api_key, **kwargs)
     cls._instances[key] = new_instance
     return new_instance

if __name__ == "__main__":
  from shared.config import OPENAI_API_KEY
  llm1 = LLMProvider.get_llm(model="gpt-4o-mini", api_key=OPENAI_API_KEY, xpto=23)
  llm2 = LLMProvider.get_llm(model="gpt-4o-mini", api_key=OPENAI_API_KEY, xpto=23)
  print(f"LLM instances are the same: {llm1 is llm2}")