import json
from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openevals.prompts import RAG_GROUNDEDNESS_PROMPT
from pydantic import BaseModel
from app.config import SETTINGS_FILE
from app.llm import prompt_guard_llm, safeguard_llm, strong_llm

MAX_QUESTION_LENGTH = 1000

# Which guards are switched on, and the numbers they use. Saved in data/settings.json.
DEFAULT_SETTINGS = {
    "guards": {
        "length": {"enabled": True},
        "prompt_injection_input": {"enabled": True},
        "safety_input": {"enabled": True},
        "prompt_injection_document": {"enabled": True},
        "pii_secrets_output": {"enabled": True},
        "hallucination_output": {"enabled": True},
        "safety_output": {"enabled": True},
        "citation_output": {"enabled": True},
    },
    "thresholds": {"dense_min_score": 0.55, "prompt_guard_min_score": 0.5},
}

# The rules we give the safety model. It reads the text and answers in JSON.
SAFETY_RULES = """You check a text for two kinds of violations.
1. toxicity: rude, hateful, abusive or threatening language.
2. code_injection: the text asks for, or contains, shell commands, SQL or scripts that are meant to run or to cause harm (for example deleting files, dropping tables, os.system).
Normal questions about security, AI or software are NOT violations.
Reply with JSON only, in exactly this form:
{"violation": 0 or 1, "category": "toxicity" or "code_injection" or "none", "rationale": "one short sentence"}"""

# The safety model followed by LangChain's JSON reader: the reply becomes a Python dictionary.
safety_chain = safeguard_llm | JsonOutputParser()



class GroundednessVerdict(BaseModel):
    reasoning: str    # first, so the model explains before it decides
    grounded: bool    # true = every claim in the answer is supported by the retrieved documents


# The judge. The rules come from LangChain's openevals library (RAG_GROUNDEDNESS_PROMPT).
# The verdict must have the shape of GroundednessVerdict. We ask Groq for it with its "structured output"
# feature in strict mode: Groq then builds the reply to fit the shape, so it cannot be a bare "false".
# (Two other ways failed on Groq: a "tool call", which the model sometimes skips, and plain "JSON mode",
# where the model once replied with only the word false.)
groundedness_prompt = ChatPromptTemplate.from_messages([
    ("system", "Give your reasoning as text, and set grounded to true only if every claim in the output "
               "is supported by the context."),
    ("human", RAG_GROUNDEDNESS_PROMPT),
])
groundedness_chain = groundedness_prompt | strong_llm.with_structured_output(
    GroundednessVerdict, method="json_schema", strict=True
)


def make_pii_middleware(pii_type: str, detector: str | None = None) -> PIIMiddleware:
    """One PII check from LangChain. It replaces private data in the ANSWER with [REDACTED_<TYPE>].
    We do not check the input (the text going into the model), because the retrieved documents are part
    of it and they contain web addresses and email addresses that we want to keep."""
    return PIIMiddleware(pii_type, detector=detector, strategy="redact",
                         apply_to_input=False, apply_to_output=True)


# Built-in types of LangChain: they find these automatically.
# Our own types: a text pattern (a "regex") says what to look for.
pii_middleware = [
    make_pii_middleware("email"),
    make_pii_middleware("credit_card"),
    make_pii_middleware("ip"),
    make_pii_middleware("api_key", r"(?:gsk|hf|lsv2|sk)[-_][A-Za-z0-9_\-]{16,}"),  # Groq, HuggingFace, LangSmith, OpenAI keys
    make_pii_middleware("password", r"(?i)password\s*[:=]\s*\S+"),                    # for example: password: abc123
]


def make_result(guard_name: str, action: str, detail: str = "") -> dict:
    """Every guard returns this same small dictionary. action is allow, block, drop or redact.
    file and page are only filled for the document guard (which document the result is about).
    For all other guards they stay empty, so every result has the same fields."""
    return {"guard": guard_name, "action": action, "detail": detail, "file": "", "page": None}


def load_settings() -> dict:
    """Reads data/settings.json. If the file does not exist yet, it is created with the default values."""
    if not SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, "w", encoding="utf-8") as settings_file:
            json.dump(DEFAULT_SETTINGS, settings_file, indent=2)
        return DEFAULT_SETTINGS

    with open(SETTINGS_FILE, encoding="utf-8") as settings_file:
        settings = json.load(settings_file)

    # A file written by an older version may miss guards or numbers that were added later. Add them.
    for guard_name, guard_setting in DEFAULT_SETTINGS["guards"].items():
        if guard_name not in settings["guards"]:
            settings["guards"][guard_name] = guard_setting
    for threshold_name, value in DEFAULT_SETTINGS["thresholds"].items():
        if threshold_name not in settings["thresholds"]:
            settings["thresholds"][threshold_name] = value
    return settings


def save_settings(settings: dict) -> None:
    """Writes the settings to data/settings.json."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as settings_file:
        json.dump(settings, settings_file, indent=2)


def run_guard(settings: dict, guard_name: str, check_function, *arguments) -> dict:
    """Runs one guard, unless it is switched off in the settings. The arguments are passed on to check_function."""
    if not settings["guards"][guard_name]["enabled"]:
        return make_result(guard_name, "allow", "disabled")
    return check_function(*arguments)


def check_length(question: str) -> dict:
    if len(question.strip()) == 0:
        return make_result("length", "block", "The question is empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        return make_result("length", "block", f"The question is longer than {MAX_QUESTION_LENGTH} characters.")
    return make_result("length", "allow")


def get_prompt_guard_score(text: str) -> float:
    """Asks Llama Prompt Guard 2 how likely the text is a trick: 0 = normal, 1 = trick. It reads about 512 tokens."""
    reply = prompt_guard_llm.invoke(text)
    reply_text = str(reply.content).strip()
    try:
        return float(reply_text)
    except ValueError:
        # We did not know the exact reply format when this was written. Show the real reply.
        raise ValueError(f"Unexpected reply from Prompt Guard: {reply_text!r}")


def check_prompt_injection_input(question: str, min_score: float) -> dict:
    score = get_prompt_guard_score(question)
    if score >= min_score:
        return make_result("prompt_injection_input", "block", f"score {score:.2f}")
    return make_result("prompt_injection_input", "allow", f"score {score:.2f}")


def check_prompt_injection_document(document_text: str, min_score: float) -> dict:
    score = get_prompt_guard_score(document_text)
    if score >= min_score:
        # The user did nothing wrong, so we only drop this document. The request carries on.
        return make_result("prompt_injection_document", "drop", f"score {score:.2f}")
    return make_result("prompt_injection_document", "allow", f"score {score:.2f}")


def filter_documents(settings: dict, retrieved_docs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Checks every retrieved document and drops the ones with hidden instructions.
    Returns the documents to keep, and the guard results to report."""
    if not settings["guards"]["prompt_injection_document"]["enabled"]:
        return retrieved_docs, [make_result("prompt_injection_document", "allow", "disabled")]

    min_score = settings["thresholds"]["prompt_guard_min_score"]
    kept_docs = []
    guard_results = []
    for retrieved_doc in retrieved_docs:
        result = check_prompt_injection_document(retrieved_doc["text"], min_score)

        # Result is in the following format, for example:
        # {"guard": "prompt_injection_document", "action": "allow", "detail": "score 0.00"}
        # We are adding the file and page to it, so that we can identify the result belongs to which document. The final result will look like this.
        # {"guard": "prompt_injection_document", "action": "allow", "detail": "score 0.00", "file": "nist-ai-rmf-1.0.pdf", "page": 8}
        result["file"] = retrieved_doc["file"]
        result["page"] = retrieved_doc["page"]

        guard_results.append(result)
        if result["action"] != "drop":
            kept_docs.append(retrieved_doc)

    return kept_docs, guard_results


def documents_to_text(retrieved_docs: list[dict]) -> str:
    """Puts the retrieved documents into one text, numbered like the answer cites them.
    Example: "[1] first document text\n\n[2] second document text"."""
    numbered_texts = []
    for doc_number, retrieved_doc in enumerate(retrieved_docs, start=1):
        text = retrieved_doc["text"]
        numbered_texts.append(f"[{doc_number}] {text}")
    return "\n\n".join(numbered_texts)


def check_hallucination(answer: str, retrieved_docs: list[dict]) -> dict:
    """Asks the judge if the answer says only things that the retrieved documents support."""
    if answer.strip().startswith("I don't know"):
        return make_result("hallucination_output", "allow", "no claims to check")

    verdict = groundedness_chain.invoke({"context": documents_to_text(retrieved_docs), "outputs": answer})
    if verdict.grounded:
        return make_result("hallucination_output", "allow", "supported by the documents")
    return make_result("hallucination_output", "block", verdict.reasoning[:300])


def check_pii_redaction(settings: dict, answer: str) -> dict:
    """The PII middleware hides private data inside the answer step. This only reports what it did."""
    if not settings["guards"]["pii_secrets_output"]["enabled"]:
        return make_result("pii_secrets_output", "allow", "disabled")
    if "[REDACTED_" in answer:
        return make_result("pii_secrets_output", "redact", "private data in the answer was replaced by [REDACTED_...]")
    return make_result("pii_secrets_output", "allow")


def check_safety(text: str, guard_name: str) -> dict:
    """guard_name is "safety_input" (for the question) or "safety_output" (for the answer)."""
    messages = [SystemMessage(content=SAFETY_RULES), HumanMessage(content=text)]
    verdict = safety_chain.invoke(messages)
    if int(verdict["violation"]) == 1:
        return make_result(guard_name, "block", f"{verdict['category']}: {verdict['rationale']}")
    return make_result(guard_name, "allow")


def check_citations(answer: str, document_count: int) -> dict:
    """Every source number like [3] in the answer must point to a real retrieved document."""
    if answer.strip().startswith("I don't know"):
        return make_result("citation_output", "allow")

    cited_any = False
    for number in range(1, 100):
        if f"[{number}]" in answer:
            cited_any = True
            if number > document_count:
                return make_result("citation_output", "block", f"The answer cites [{number}], which does not exist.")

    if not cited_any:
        return make_result("citation_output", "block", "The answer cites no source.")
    return make_result("citation_output", "allow")
