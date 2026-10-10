import re

FALLACY_RULES = {
    "Ad Hominem": [r"\bidiot\b", r"\bstupid\b", r"\bdumb\b", r"\bmoron\b", r"\buneducated\b", r"\byou are just\b", r"\byou're just\b"],
    "Straw Man": [r"\bso you are saying\b", r"\bso you're saying\b", r"\byou basically think\b", r"\baccording to you\b"],
    "False Dilemma": [r"\beither .* or\b", r"\bonly two options\b", r"\bthere are only two\b", r"\byou are either\b"],
    "Hasty Generalization": [r"\ball .* are\b", r"\ball .* is\b", r"\beveryone .* is\b", r"\bpeople always\b", r"\bpeople never\b"],
    "Slippery Slope": [r"\bif we allow .* then\b", r"\bif we let .* then\b", r"\bthis will lead to\b", r"\bnext thing you know\b"],
    "Bandwagon Appeal": [r"\beveryone believes\b", r"\beverybody believes\b", r"\bmost people believe\b", r"\beveryone agrees\b", r"\bpopular opinion\b"],
    "Appeal to Authority": [r"\ban expert says\b", r"\bexperts say\b", r"\baccording to .* expert\b", r"\bscientists say\b"],
    "Appeal to Emotion": [r"\bthink about the children\b", r"\bhow would you feel\b", r"\byou should be ashamed\b", r"\bpeople will suffer\b", r"\bthis is heartbreaking\b", r"\bthis is terrifying\b"],
    "Circular Reasoning": [r"\bbecause it is true\b", r"\bit is true because\b", r"\bthat proves that it is true\b"],
    "Red Herring": [r"\bwhat about\b", r"\blet us talk about something else\b", r"\bthat's not the point\b"],
    "False Cause": [r"\btherefore .* caused\b", r"\bbecause .* happened .* then\b", r"\bsince .* happened .* caused\b"],
}

def detect_fallacies(argument):
    text = str(argument).lower()
    results = []
    for name, patterns in FALLACY_RULES.items():
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text))
        if matches:
            results.append({
                "name": name,
                "confidence": "Possible",
                "matches": len(matches),
                "description": f"A linguistic pattern associated with {name} was detected."
            })
    return results
