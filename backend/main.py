import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .search_engine import SearchEngine
from .boolean_model import QueryError

DEFAULT_DATASET=Path(__file__).resolve().parents[1]/"Dataset for autism copy"
engine=SearchEngine(os.getenv("AUTISM_DATASET_DIR",str(DEFAULT_DATASET)))
app=FastAPI(title="Autism IR Search API",version="1.0.0",description="Boolean, TF-IDF cosine, query-likelihood, BIM, and BM25 search over the supplied multimodal sample dataset.")
app.add_middleware(CORSMiddleware,allow_origins=[x.strip() for x in os.getenv("CORS_ORIGINS","http://localhost:5173").split(",")],allow_methods=["*"],allow_headers=["*"])

class SearchRequest(BaseModel):
    query:str
    modality:str="all"
    top_k:int=Field(10,ge=1,le=100)

class BIMSearchRequest(SearchRequest):
    relevant_ids:list[str]=Field(default_factory=list)
    non_relevant_ids:list[str]=Field(default_factory=list)

@app.get("/health")
def health(): return {"status":"ok","documents":len(engine.documents)}

@app.get("/dataset/stats")
def stats(): return engine.stats()

@app.post("/search/boolean")
def boolean_search(req:SearchRequest):
    try: results=engine.boolean.search(req.query,req.modality)
    except QueryError as e: raise HTTPException(400,str(e))
    return {"query":req.query,"model":"boolean","total_results":len(results),"results":results[:req.top_k]}

@app.post("/search/vector")
def vector_search(req:SearchRequest):
    try: results=engine.vector.search(req.query,req.top_k,req.modality)
    except ValueError as e: raise HTTPException(400,str(e))
    return {"query":req.query,"model":"vector_space","total_results":len(results),"results":results}

@app.post("/search/probabilistic")
def probabilistic_search(req:SearchRequest):
    try: results=engine.probabilistic.search(req.query,100,req.modality)
    except ValueError as e: raise HTTPException(400,str(e))
    return {"query":req.query,"model":"probabilistic_query_likelihood","total_results":len(results),"results":results[:req.top_k]}

@app.post("/search/bim")
def bim_search(req:BIMSearchRequest):
    try:
        results=engine.bim.search(req.query,100,req.modality,req.relevant_ids,req.non_relevant_ids)
    except ValueError as e:
        raise HTTPException(400,str(e))
    return {"query":req.query,"model":"binary_independence","feedback_used":bool(req.relevant_ids or req.non_relevant_ids),"total_results":len(results),"results":results[:req.top_k]}

@app.post("/search/bm25")
def bm25_search(req:SearchRequest):
    try: results=engine.bm25.search(req.query,100,req.modality)
    except ValueError as e: raise HTTPException(400,str(e))
    return {"query":req.query,"model":"bm25","total_results":len(results),"results":results[:req.top_k]}

@app.get("/document/{document_id}")
def document(document_id:str):
    result=engine.by_id.get(document_id)
    if not result: raise HTTPException(404,"Document not found.")
    return result

@app.get("/file/{document_id}/{modality}")
def serve_file(document_id:str,modality:str):
    d=engine.by_id.get(document_id)
    if not d or modality not in d["files"]: raise HTTPException(404,"File not found.")
    info=d["files"][modality]; path=info.get("path")
    if not path or not Path(path).is_file(): raise HTTPException(404,"File is unavailable in the dataset.")
    return FileResponse(path,filename=info["filename"])
