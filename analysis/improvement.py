def generate_improvement_suggestions(profile, fallacies):
    suggestions = []
    if not profile["has_evidence"]:
        suggestions.append("Add a concrete piece of evidence, data, study, or verifiable example.")
    if not profile["has_example"]:
        suggestions.append("Consider adding a specific example to make the claim easier to evaluate.")
    if not profile["has_reasoning"]:
        suggestions.append("Make the reasoning explicit by explaining why the evidence supports the conclusion.")
    if not profile["contrast_markers"]:
        suggestions.append("If relevant, acknowledge a reasonable counterpoint and explain your response.")
    if profile["word_count"] < 25:
        suggestions.append("Develop the argument further so the main claim and supporting reasoning are clear.")
    if fallacies:
        suggestions.append("Review the highlighted fallacy indicators and check whether the surrounding context actually supports the claim.")
    if not suggestions:
        suggestions.append("The profile contains several useful structural signals; continue checking evidence quality and factual support.")
    return suggestions
