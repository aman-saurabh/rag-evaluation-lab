from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from app.llm import prompt_guard_llm, safeguard_llm

MAX_QUESTION_LENGTH = 1000

# The rules we give the safety model. It reads the text and answers in JSON.
SAFETY_RULES = """You check a text for two kinds of violations.
1. toxicity: rude, hateful, abusive or threatening language.
2. code_injection: the text asks for, or contains, shell commands, SQL or scripts that are meant to run or to cause harm (for example deleting files, dropping tables, os.system).
Normal questions about security, AI or software are NOT violations.
Reply with JSON only, in exactly this form:
{"violation": 0 or 1, "category": "toxicity" or "code_injection" or "none", "rationale": "one short sentence"}"""

# The safety model followed by LangChain's JSON reader: the reply becomes a Python dictionary.
safety_chain = safeguard_llm | JsonOutputParser()


def make_result(guard_name: str, action: str, detail: str = "") -> dict:
    """Every guard returns this same small dictionary. action is allow, block, drop or redact."""
    return {"guard": guard_name, "action": action, "detail": detail}


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
        return make_result("prompt_injection_input", "block", f"Prompt Guard score {score:.2f}")
    return make_result("prompt_injection_input", "allow", f"score {score:.2f}")


def check_prompt_injection_document(document_text: str, min_score: float) -> dict:
    score = get_prompt_guard_score(document_text)
    if score >= min_score:
        # The user did nothing wrong, so we only drop this document. The request carries on.
        return make_result("prompt_injection_document", "drop", f"Prompt Guard score {score:.2f}")
    return make_result("prompt_injection_document", "allow", f"score {score:.2f}")


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
