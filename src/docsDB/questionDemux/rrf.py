from langchain_core.documents import Document
from docsDB.questionDemux import RRF_PARAM

def reciprocal_rank_fusion(results_groups: list[list[Document]], k=RRF_PARAM) -> list[tuple[Document, float]]:
  """ Reciprocal_rank_fusion that takes multiple groups of 
      ranked documents and an optional parameter k used in 
      the Reciprocal Rank Fusion (RRF) formula """
  
  fused_scores: dict[str, float] = {}
  documents: dict[str, Document] = {}
  for results_group in results_groups:
    for rank, doc in enumerate(results_group):
      doc_id = doc.id or doc.metadata.get("id") or doc.metadata.get("doc_id")
      documents[doc_id] = doc
      fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)

  return [(documents[doc_id], score)
          for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
