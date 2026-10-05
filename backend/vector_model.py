import math
from collections import Counter
from .preprocessing import tokenize

class VectorModel:
    def __init__(self, documents):
        self.documents=documents
        self.tf=[]; df=Counter()
        for d in documents:
            c=Counter(d["tokens"]); self.tf.append(c); df.update(c.keys())
        n=len(documents)
        self.idf={t:math.log((1+n)/(1+freq))+1 for t,freq in df.items()}
        self.vectors=[self._weighted(c) for c in self.tf]
        self.norms=[math.sqrt(sum(v*v for v in x.values())) for x in self.vectors]
    def _weighted(self,c):
        return {t:(1+math.log(n))*self.idf[t] for t,n in c.items() if n and t in self.idf}
    def search(self,query,top_k=10,modality="all"):
        terms=tokenize(query)
        if not terms: raise ValueError("Enter a search query containing searchable terms.")
        if top_k<1: raise ValueError("top_k must be at least 1.")
        q=self._weighted(Counter(terms)); norm=math.sqrt(sum(v*v for v in q.values()))
        ranked=[]
        for d,v,dn in zip(self.documents,self.vectors,self.norms):
            if modality and modality.lower() not in {"all","sample"} and not d["files"].get(modality.lower(),{}).get("exists"): continue
            dot=sum(w*v.get(t,0) for t,w in q.items()); score=dot/(norm*dn) if norm and dn else 0
            if score>0: ranked.append((score,d))
        ranked.sort(key=lambda x:(-x[0],x[1]["document_id"]))
        return [{**d,"score":score,"matched_terms":sorted(set(terms)&set(d["tokens"]))} for score,d in ranked[:top_k]]
