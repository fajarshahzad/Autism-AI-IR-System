"""Metrics for an explicitly supplied query -> relevant document ID mapping."""
def evaluate(retrieved_ids, relevant_ids, k=None):
    relevant=set(relevant_ids); retrieved=list(retrieved_ids if k is None else retrieved_ids[:k]); found=set(retrieved)&relevant
    precision=len(found)/len(retrieved) if retrieved else 0.0
    recall=len(found)/len(relevant) if relevant else 0.0
    return {"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "precision_at_k":precision if k is not None else None,"recall_at_k":recall if k is not None else None}

def evaluate_queries(engine, judgments, model="vector_space", top_k=10):
    reports={}
    for query, relevant_ids in judgments.items():
        if model=="vector_space": results=engine.vector.search(query,top_k)
        elif model=="probabilistic_query_likelihood": results=engine.probabilistic.search(query,top_k)
        elif model=="binary_independence": results=engine.bim.search(query,top_k)
        elif model=="boolean": results=engine.boolean.search(query)
        else: raise ValueError(f"Unknown retrieval model: {model}")
        reports[query]=evaluate([x["document_id"] for x in results],relevant_ids,top_k)
    return reports
