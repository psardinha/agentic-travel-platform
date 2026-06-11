from docsDB.load_db import build_collection

_collection = None

def get_collection():
  global _collection
  if _collection is None:
    _collection = build_collection()
  return _collection
