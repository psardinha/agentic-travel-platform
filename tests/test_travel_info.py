from shared.config import OPENAI_API_KEY
from core.llm_provider import LLMProvider
from docsDB.store import get_collection
from travel_info_agent.agent import docs_retriver

def test_search_travel_info():
  docsCollection = get_collection()
  llm = LLMProvider.get_llm(api_key=OPENAI_API_KEY)
  rag_fusion_retrieval_chain = docs_retriver(llm, docsCollection)
  query = "Are there nice beaches in Cornwall?"
  NUM_TOP_DOCS = 1
  docs = rag_fusion_retrieval_chain.invoke(query)
  top = docs[:NUM_TOP_DOCS] if isinstance(docs, list) else docs
  data_retrieved = "\n---\n".join(d[0].page_content for d in top)
  assert len(data_retrieved) > 0
