import re

MARKERS = {
    "reasoning": ["because", "therefore", "thus", "hence", "since", "as a result", "consequently", "which means"],
    "evidence": ["evidence", "study", "studies", "research", "data", "survey", "statistics", "statistic", "report", "according", "source"],
    "example": ["for example", "for instance", "such as", "e.g."],
    "contrast": ["however", "although", "though", "but", "while", "whereas", "despite"],
    "conclusion": ["therefore", "thus", "hence", "overall", "in conclusion", "ultimately"],
}

def _count(text, phrases):
    text = text.lower()
    return sum(text.count(p) for p in phrases)

def get_argument_profile(argument):
    text = str(argument).strip()
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    wc = len(words)
    unique = len(set(w.lower() for w in words))
    profile = {
        "word_count": wc,
        "unique_word_count": unique,
        "unique_ratio": round(unique / wc, 3) if wc else 0,
        "sentence_count": len(sentences),
        "average_sentence_length": round(wc / len(sentences), 2) if sentences else 0,
        "reasoning_markers": _count(text, MARKERS["reasoning"]),
        "evidence_markers": _count(text, MARKERS["evidence"]),
        "example_markers": _count(text, MARKERS["example"]),
        "contrast_markers": _count(text, MARKERS["contrast"]),
        "conclusion_markers": _count(text, MARKERS["conclusion"]),
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "number_count": len(re.findall(r"\b\d+(?:\.\d+)?\b", text)),
        "has_reasoning": _count(text, MARKERS["reasoning"]) > 0,
        "has_evidence": _count(text, MARKERS["evidence"]) > 0,
        "has_example": _count(text, MARKERS["example"]) > 0,
    }
    return profile
