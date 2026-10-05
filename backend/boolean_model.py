import re
from .preprocessing import tokenize

class QueryError(ValueError): pass

class BooleanModel:
    def __init__(self, documents):
        self.documents = {d["document_id"]: d for d in documents}
        self.universe = set(self.documents)
        self.index = {}
        for d in documents:
            for term in set(d["tokens"]):
                self.index.setdefault(term, set()).add(d["document_id"])

    def search(self, query, modality="all"):
        if not query or not query.strip(): raise QueryError("Enter a Boolean query.")
        parts = re.findall(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+", query, flags=re.I)
        if not parts: raise QueryError("Enter a Boolean query.")
        output=[]; expect_operand=True
        for p in parts:
            u=p.upper()
            if u in {"AND","OR"}:
                if expect_operand: raise QueryError(f"Unexpected operator: {p}")
                output.append(u); expect_operand=True
            elif u=="NOT":
                if not expect_operand: raise QueryError("NOT must follow AND/OR or start the query.")
                output.append("NOT")
            elif p=="(": output.append(p)
            elif p==")": output.append(p)
            else:
                terms=tokenize(p)
                if not terms: raise QueryError(f"Invalid search term: {p}")
                # Labels such as mild_asd normalize to adjacent searchable terms.
                for term in terms:
                    if not expect_operand: output.append("AND")
                    output.append(term); expect_operand=False
        if expect_operand: raise QueryError("Query ends with an operator or NOT.")
        # Shunting-yard; NOT > AND > OR.
        prec={"OR":1,"AND":2,"NOT":3}; out=[]; ops=[]
        for t in output:
            if t in prec:
                while ops and ops[-1] in prec and prec[ops[-1]]>=prec[t]: out.append(ops.pop())
                ops.append(t)
            elif t=="(": ops.append(t)
            elif t==")":
                while ops and ops[-1]!="(": out.append(ops.pop())
                if not ops: raise QueryError("Unbalanced parentheses.")
                ops.pop()
            else: out.append(t)
        if "(" in ops: raise QueryError("Unbalanced parentheses.")
        out.extend(reversed(ops)); stack=[]
        for t in out:
            if t=="NOT":
                if not stack: raise QueryError("NOT has no operand.")
                stack.append(self.universe-stack.pop())
            elif t in {"AND","OR"}:
                if len(stack)<2: raise QueryError("Operator has too few operands.")
                b,a=stack.pop(),stack.pop(); stack.append(a&b if t=="AND" else a|b)
            else: stack.append(set(self.index.get(t,set())))
        if len(stack)!=1: raise QueryError("Invalid Boolean query.")
        result=stack.pop()
        if modality and modality.lower() not in {"all","sample"}:
            key=modality.lower(); result={i for i in result if self.documents[i]["files"].get(key,{}).get("exists")}
        terms=set(tokenize(re.sub(r"\b(?:AND|OR|NOT)\b|[()]", " ", query, flags=re.I)))
        return [{**self.documents[i], "matched_terms": sorted(terms & set(self.documents[i]["tokens"]))} for i in sorted(result)]
