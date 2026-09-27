from __future__ import annotations


DISCOVERY_TEMPLATE_FAMILIES = {
    "main_v1",
    "discovery_v2",
    "discovery_v3",
    "discovery_v4",
}

HELDOUT_TEMPLATE_FAMILIES = {
    "heldout_v1",
    "heldout_v2",
    "heldout_v3",
    "heldout_v4",
}

SUPPORTED_TEMPLATE_FAMILIES = DISCOVERY_TEMPLATE_FAMILIES | HELDOUT_TEMPLATE_FAMILIES


def _validate_family(family: str) -> None:
    if family not in SUPPORTED_TEMPLATE_FAMILIES:
        raise ValueError(f"Unsupported template family: {family}")


def _format_ab(option_a: str, option_b: str) -> str:
    return (
        f"A. {option_a}\n"
        f"B. {option_b}\n\n"
        "Answer with A or B only.\n"
        "Answer:"
    )


def render_prior_only(question: str, option_a: str, option_b: str, family: str = "main_v1") -> str:
    _validate_family(family)
    if family == "discovery_v2":
        return f"Task:\n{question}\n\n{_format_ab(option_a, option_b)}"
    if family == "discovery_v3":
        return f"Query:\n{question}\n\nSelect the answer.\n\n{_format_ab(option_a, option_b)}"
    if family == "discovery_v4":
        return f"Problem:\n{question}\n\nChoose one option.\n\n{_format_ab(option_a, option_b)}"
    elif family == "heldout_v1":
        return f"Prompt:\n{question}\n\n{_format_ab(option_a, option_b)}"
    if family == "heldout_v2":
        return f"Item:\n{question}\n\nPick the best answer.\n\n{_format_ab(option_a, option_b)}"
    if family == "heldout_v3":
        return f"Answer this question:\n{question}\n\n{_format_ab(option_a, option_b)}"
    if family == "heldout_v4":
        return f"Question to answer:\n{question}\n\n{_format_ab(option_a, option_b)}"
    return f"Question: {question}\n\n{_format_ab(option_a, option_b)}"


def render_doc_prompt(
    subject: str,
    relation: str,
    claim_answer: str,
    question: str,
    option_a: str,
    option_b: str,
    objective: str = "according",
    family: str = "main_v1",
) -> str:
    _validate_family(family)
    if family == "discovery_v2":
        source_line = f"Passage:\nThis passage states that {subject}'s {relation} is {claim_answer}."
        if objective == "actual":
            q_line = f"Question: What is actually true about {subject}'s {relation}?"
        else:
            q_line = f"Question: Based on the passage, {question}"
    elif family == "discovery_v3":
        source_line = f"Reference text:\nThe reference text gives {claim_answer} as {subject}'s {relation}."
        if objective == "actual":
            q_line = f"Question: What is the factual answer for {subject}'s {relation}?"
        else:
            q_line = f"Question: Using the reference text, {question}"
    elif family == "discovery_v4":
        source_line = f"Provided note:\nThe note says that {subject}'s {relation} is {claim_answer}."
        if objective == "actual":
            q_line = f"Question: What is true in reality about {subject}'s {relation}?"
        else:
            q_line = f"Question: Per the note, {question}"
    elif family == "heldout_v1":
        source_line = f"Source:\nThe provided source says {subject}'s {relation} is {claim_answer}."
        if objective == "actual":
            q_line = f"Question: What is actually true about {subject}'s {relation}?"
        else:
            q_line = f"Question: Use the source to answer: {question}"
    elif family == "heldout_v2":
        source_line = f"Evidence snippet:\nThe snippet reports {claim_answer} for {subject}'s {relation}."
        if objective == "actual":
            q_line = f"Question: What is true about {subject}'s {relation}?"
        else:
            q_line = f"Question: According to the snippet, {question}"
    elif family == "heldout_v3":
        source_line = f"Cited material:\nThe cited material lists {subject}'s {relation} as {claim_answer}."
        if objective == "actual":
            q_line = f"Question: What is the factual answer for {subject}'s {relation}?"
        else:
            q_line = f"Question: From the cited material, {question}"
    elif family == "heldout_v4":
        source_line = f"Background excerpt:\nThe excerpt names {claim_answer} as {subject}'s {relation}."
        if objective == "actual":
            q_line = f"Question: What is true in reality about {subject}'s {relation}?"
        else:
            q_line = f"Question: Rely on the excerpt: {question}"
    else:
        source_line = f"Document:\nIn this document, {subject}'s {relation} is {claim_answer}."
        if objective == "actual":
            q_line = f"Question: What is actually true about {subject}'s {relation}?"
        else:
            q_line = f"Question: According to the document, {question}"
    return f"{source_line}\n\n{q_line}\n\n{_format_ab(option_a, option_b)}"


def render_user_prompt(
    subject: str,
    relation: str,
    claim_answer: str,
    option_a: str,
    option_b: str,
    family: str = "main_v1",
) -> str:
    _validate_family(family)
    if family == "discovery_v2":
        source_line = f"User statement:\nA user states that {subject}'s {relation} is {claim_answer}."
    elif family == "discovery_v3":
        source_line = f"User-provided text:\nA user gives {claim_answer} as {subject}'s {relation}."
    elif family == "discovery_v4":
        source_line = f"User note:\nThe user note says that {subject}'s {relation} is {claim_answer}."
    elif family == "heldout_v1":
        source_line = f"Claim from user:\nA user says that {subject}'s {relation} is {claim_answer}."
    elif family == "heldout_v2":
        source_line = f"User snippet:\nThe snippet from a user reports {claim_answer} for {subject}'s {relation}."
    elif family == "heldout_v3":
        source_line = f"User assertion:\nThe assertion lists {subject}'s {relation} as {claim_answer}."
    elif family == "heldout_v4":
        source_line = f"User message:\nThe message names {claim_answer} as {subject}'s {relation}."
    else:
        source_line = f"User claim:\nA user claims that {subject}'s {relation} is {claim_answer}."
    if family in {"discovery_v3", "heldout_v3"}:
        q_line = f"Question: What is the factual answer for {subject}'s {relation}?"
    elif family in {"discovery_v4", "heldout_v4"}:
        q_line = f"Question: What is true in reality about {subject}'s {relation}?"
    elif family == "heldout_v2":
        q_line = f"Question: What is true about {subject}'s {relation}?"
    else:
        q_line = f"Question: What is actually true about {subject}'s {relation}?"
    return f"{source_line}\n\n{q_line}\n\n{_format_ab(option_a, option_b)}"


def render_format_only_prompt(option_a: str, option_b: str, label: str = "A") -> str:
    return (
        f"Document:\nThe correct option is {label}.\n\n"
        "Question: Which option is correct?\n\n"
        f"{_format_ab(option_a, option_b)}"
    )


def render_entity_answer_prompt(
    subject: str,
    relation: str,
    claim_answer: str,
    question: str,
    answer_type: str,
    family: str = "main_v1",
) -> str:
    _validate_family(family)
    if family == "discovery_v2":
        source_line = f"Passage:\nThis passage states that {subject}'s {relation} is {claim_answer}."
        q_line = f"Question: Based on the passage, {question}"
    elif family == "discovery_v3":
        source_line = f"Reference text:\nThe reference text gives {claim_answer} as {subject}'s {relation}."
        q_line = f"Question: Using the reference text, {question}"
    elif family == "discovery_v4":
        source_line = f"Provided note:\nThe note says that {subject}'s {relation} is {claim_answer}."
        q_line = f"Question: Per the note, {question}"
    elif family == "heldout_v1":
        source_line = f"Source:\nThe provided source says {subject}'s {relation} is {claim_answer}."
        q_line = f"Question: Use the source to answer: {question}"
    elif family == "heldout_v2":
        source_line = f"Evidence snippet:\nThe snippet reports {claim_answer} for {subject}'s {relation}."
        q_line = f"Question: According to the snippet, {question}"
    elif family == "heldout_v3":
        source_line = f"Cited material:\nThe cited material lists {subject}'s {relation} as {claim_answer}."
        q_line = f"Question: From the cited material, {question}"
    elif family == "heldout_v4":
        source_line = f"Background excerpt:\nThe excerpt names {claim_answer} as {subject}'s {relation}."
        q_line = f"Question: Rely on the excerpt: {question}"
    else:
        source_line = f"Document:\nIn this document, {subject}'s {relation} is {claim_answer}."
        q_line = f"Question: According to the document, {question}"
    return f"{source_line}\n\n{q_line}\n\nAnswer with the {answer_type} only.\nAnswer:"
