from .data_loader import load_dataset
from .document_builder import build_documents
from .boolean_model import BooleanModel
from .vector_model import VectorModel
from .probabilistic_model import ProbabilisticModel
from .bim_model import BinaryIndependenceModel

class SearchEngine:
    def __init__(self, dataset_dir):
        self.root, rows=load_dataset(dataset_dir)
        self.documents=build_documents(self.root,rows)
        self.by_id={d["document_id"]:d for d in self.documents}
        self.boolean=BooleanModel(self.documents)
        self.vector=VectorModel(self.documents)
        self.probabilistic=ProbabilisticModel(self.documents)
        self.bim=BinaryIndependenceModel(self.documents)

    def stats(self):
        labels=sorted({str(d["label"]) for d in self.documents if d["label"]})
        counts={m:sum(1 for d in self.documents if d["files"][m]["exists"]) for m in ("image","motion","physio","voice")}
        types=sorted({f["file_type"] for d in self.documents for f in d["files"].values() if f["file_type"]})
        return {"total_documents":len(self.documents),"modalities":counts,"labels":labels,"file_types":types,
                "available_modalities":[m for m,n in counts.items() if n],"dataset_path":str(self.root),
                "missing_indexed_files":sum(not f["indexed_path_exists"] for d in self.documents for f in d["files"].values())}
