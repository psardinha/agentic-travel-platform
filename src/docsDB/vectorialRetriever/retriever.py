from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from docsDB.vectorialRetriever import METADATA_FIELD_INFO, DEF_K
from docsDB.vectorialRetriever.DestinationSearch import DestinationSearch
from langchain_classic.chains.query_constructor.ir import (Comparator, Comparison, Operation, Operator, StructuredQuery)
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.prompts import ChatPromptTemplate


def getDefaultRetriever(collection, filter = None, k =DEF_K):
  return collection.as_retriever(search_kwargs={'k':k, 'filter': filter})

def getDefaultRetriever_with_scores (collection, query, filter=None, k=DEF_K):
  return collection.similarity_search_with_score(query, k=k, filter=filter)

def getSelfQueryRetriever(llm, collection, query, k=DEF_K,):
  #Automatically generate the metadata filter using the SelfQueryRetriever.
  return SelfQueryRetriever.from_llm(llm, collection, query, METADATA_FIELD_INFO, search_kwargs={"k": k}, verbose=True)

def build_filter(destination_search: DestinationSearch):
  comparisons = []
  destination = destination_search.destination 
  region = destination_search.region 

  if (destination is None or destination == '') and (region is None or region == ''):
    return None
  if destination and destination != '': 
    comparisons.append(Comparison(comparator=Comparator.EQ, attribute="destination", value=destination))
  if region and region != '': 
    comparisons.append(Comparison(comparator=Comparator.EQ, attribute="region", value=region))    

  search_filter = Operation(operator=Operator.AND, arguments=comparisons) 
  chroma_filter = ChromaTranslator().visit_operation(search_filter) 
  return chroma_filter

def get_structured_query(llm, question):
  system_message = ("You are an expert at converting user "
                    "questions into vector database queries.\n"
                    "You have access to a database of tourist destinations.\n"
                    "Given a question, return a database query optimized "
                    "to retrieve the most relevant results.\n"
                    "If there are acronyms or words you are not familiar with, "
                    "do not try to rephrase them.")  
  prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", "{question}")])
  structured_llm = llm.with_structured_output(DestinationSearch, method="function_calling")
  return (prompt | structured_llm).invoke(question)

def build_query_and_filter(llm, question):
  structured_query = get_structured_query(llm, question)
  search_filter = build_filter(structured_query)
  return structured_query, search_filter

def get_retriever_with_metadata_inference(llm, collection, question, k=DEF_K):
  structured_query, search_filter = build_query_and_filter(llm, question)
  #print(f"Structured query generated: {structured_query}")
  #print(f"Search filter built: {search_filter}")
  return collection.as_retriever(search_kwargs={'k':k, 'filter': search_filter})

def retrieve_with_metadata_and_scores(llm, collection, question, k=DEF_K):
  structured_query, search_filter = build_query_and_filter(llm, question)
  #print(f"Structured query generated: {structured_query}")
  #print(f"Search filter built: {search_filter}")
  return collection.similarity_search_with_score(question, k=k, filter=search_filter)

if __name__ == "__main__":
  from core.llm_provider import LLMProvider
  from shared.config import OPENAI_API_KEY
  llm = LLMProvider.get_llm(api_key=OPENAI_API_KEY)

  from docsDB.store import get_collection
  collection = get_collection()
  
  print("\n\nDefault retriever results:")
  retriever = getDefaultRetriever(collection, k=2)
  search_query = "Tell me about events or festivals in the UK town of Newquay"
  print(f"Results: {retriever.invoke(search_query)}")
  print(f"Results: {getDefaultRetriever_with_scores(collection, search_query,k=2)}")

  # print("\n\nSelf-query retriever results:")
  # search_query = "Tell me about events or festivals in the UK town of Newquay"
  # retriever = getSelfQueryRetriever(llm, collection, search_query)  
  # print(f"Results: {retriever.invoke(search_query)}")

  # print("\n\nRetriever with metadata inference results:")
  # search_query = "Tell me about events or festivals in the UK town of Newquay"
  # retriever = get_retriever_with_metadata_inference(llm, collection, search_query, k=1)
  # print(f"Results: {retriever.invoke(search_query)}")
  # search_query = "Tell me about events or festivals in the UK town of Newquay"
  # retriever = get_retriever_with_metadata_inference(llm, collection, search_query, k=1)
  # print(f"Results: {retriever.invoke(search_query)}")
  # search_query = "FgRd hTTdfg"
  # retriever = get_retriever_with_metadata_inference(llm, collection, search_query, k=1)
  # print(f"Results: {retriever.invoke(search_query)}")
