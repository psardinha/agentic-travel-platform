from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings
from shared.config import OPENAI_API_KEY, HUGGING_FACES_API_KEY

def build_HF_embedding():
  return HuggingFaceEndpointEmbeddings(model="BAAI/bge-base-en-v1.5", huggingfacehub_api_token=HUGGING_FACES_API_KEY)

def build_OPENAI_embedding():
  return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)