import os
os.environ["USER_AGENT"] = "travelAgent/0.0.1"

from langchain_chroma import Chroma
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from docsDB.ukDestinations import getDestinations_with_metadata
from docsDB import DEF_DB_COLLECTION_NAME
from docsDB.vectorialRetriever.embeddings import build_HF_embedding

def split_docs_into_chunks(html2text, splitter, docs):
  text_docs = html2text.transform_documents(docs)
  chunks = splitter.split_documents(text_docs)
  return chunks

def build_collection():
  collection = Chroma(collection_name=DEF_DB_COLLECTION_NAME,
                      embedding_function=build_HF_embedding())
  collection.reset_collection()

  html2text_transformer = Html2TextTransformer()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

  for (url, destination, region) in getDestinations_with_metadata():
    html_loader = AsyncHtmlLoader(url)
    docs =  html_loader.load()
    
    docs_with_metadata = [Document(page_content=d.page_content,
                                   metadata = {'source': url,
                                               'destination': destination,
                                               'region': region})
                          for d in docs]
  
    chunks = split_docs_into_chunks(html2text_transformer, text_splitter, docs_with_metadata)
    print(f'Importing: {destination}')
    collection.add_documents(documents=chunks)
  return collection

if __name__ == "__main__":
  collection = build_collection()
  question =  "Events or festivals  in Newquay"
  metadata_retriever = collection.as_retriever(search_kwargs={'k':2, 'filter':{'destination': 'Newquay'}})
  result_docs = metadata_retriever.invoke(question)
  print(f"Results for question: {question}")
  # for doc in result_docs:
  #   print(doc.metadata['source'])
  #   print(doc.page_content)
  metadata_retriever = collection.similarity_search_with_score(question)
  print(f"Results: {metadata_retriever}")
