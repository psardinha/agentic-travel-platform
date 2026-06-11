from langchain_core.prompts import ChatPromptTemplate
from docsDB.questionDemux.LineListOutputParser import LineListOutputParser
from docsDB.questionDemux import NUM_ALT_QUESTIONS
from docsDB.store import get_collection
from docsDB.vectorialRetriever.retriever import getDefaultRetriever

class MultiQueryChainFactory:
  _instance = None

  @classmethod
  def get_instance(cls, llm):
    if cls._instance is None:
      multi_query_gen_prompt_template = ("You are an AI language model assistant. Your task is "
                                         "to generate {num_alt_questions} different versions of the given user "
                                         "question to retrieve relevant documents from a vector "
                                         "database. By generating multiple perspectives on the "
                                         "user question, your goal is to help "
                                         "the user overcome some of the limitations of the "
                                         "distance-based similarity search.\n" 
                                         "Provide these alternative questions separated by newlines.\n"
                                         "Original question: {question}")
      prompt = ChatPromptTemplate.from_template(multi_query_gen_prompt_template).partial(num_alt_questions=NUM_ALT_QUESTIONS)
      #print(prompt)
      parser = LineListOutputParser()
      cls._instance = prompt | llm | parser
    return cls._instance
    
if __name__ == "__main__":
  from core.llm_provider import LLMProvider
  from shared.config import OPENAI_API_KEY
  llm = LLMProvider.get_llm(api_key=OPENAI_API_KEY)
  multi_query_chain = MultiQueryChainFactory.get_instance(llm)
  question = "What are some good tourist destinations in Europe?"
  # alt_questions = multi_query_chain.invoke(question)
  # print("Alternative questions generated:")
  # for idx, alt_q in enumerate(alt_questions, start=1):
  #   print(f"{idx}. {alt_q}")
  
  from docsDB.store import get_collection
  collection = get_collection()
  from docsDB.vectorialRetriever.retriever import getDefaultRetriever
  from docsDB.questionDemux.rrf import reciprocal_rank_fusion
  from langchain_core.runnables import RunnableLambda
  retriever = getDefaultRetriever(collection)
  rag_fusion_retrieval_chain = multi_query_chain | retriever.map() | reciprocal_rank_fusion | RunnableLambda(lambda x: x[0:3])
  question = "Can you give me some tips for a trip to Brighton?"
  docs = rag_fusion_retrieval_chain.invoke({"question": question})
  print("Retrieved documents:")
  for idx, doc in enumerate(docs, start=1):
    print(f"{idx}. {doc}")