import re
import numpy as np
from scipy.sparse import csr_matrix, hstack

REASONING = ["because","therefore","thus","hence","since","as a result","consequently","which means"]
EVIDENCE = ["evidence","study","studies","research","data","survey","statistics","statistic","report","according","source","example","examples"]
EXAMPLE = ["for example","for instance","such as","e.g."]
CONTRAST = ["however","although","though","but","while","whereas","despite"]
CONCLUSION = ["therefore","thus","hence","overall","in conclusion","ultimately"]

def clean_text(x):
    x = "" if x is None else str(x).lower()
    x = re.sub(r"http\S+|www\S+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def handcrafted(texts):
    out=[]
    for raw in texts:
        t=clean_text(raw)
        words=re.findall(r"\b[\w'-]+\b", t)
        sents=[s for s in re.split(r"[.!?]+",t) if s.strip()]
        wc=len(words); uniq=len(set(words))
        def c(lst): return sum(t.count(p) for p in lst)
        out.append([
            wc, uniq, uniq/wc if wc else 0, len(sents), wc/len(sents) if sents else 0,
            c(REASONING), c(EVIDENCE), c(EXAMPLE), c(CONTRAST), c(CONCLUSION),
            t.count("?"), t.count("!"), len(re.findall(r"\b\d+(?:\.\d+)?\b",t)),
            1 if len(sents)>1 else 0, 1 if c(REASONING)>0 else 0, 1 if c(EVIDENCE)>0 else 0
        ])
    return np.asarray(out, dtype=np.float32)
