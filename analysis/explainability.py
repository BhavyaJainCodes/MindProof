def explain_argument(profile, quality_score):
    notes = []
    if profile["evidence_markers"] > 0:
        notes.append("Evidence-related language was detected.")
    else:
        notes.append("No common evidence marker was detected.")
    if profile["reasoning_markers"] > 0:
        notes.append("Reasoning/causal connector language was detected.")
    else:
        notes.append("Few explicit reasoning connectors were detected.")
    if profile["example_markers"] > 0:
        notes.append("An example marker was detected.")
    else:
        notes.append("No common example marker was detected.")
    if profile["contrast_markers"] > 0:
        notes.append("Contrast/counterpoint language was detected.")
    if profile["word_count"] < 25:
        notes.append("The argument is relatively short.")
    elif profile["word_count"] > 100:
        notes.append("The argument is relatively detailed.")
    notes.append(f"Predicted dataset-quality score: {quality_score:.1f}/100.")
    return notes
