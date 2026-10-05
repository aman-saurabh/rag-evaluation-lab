import uuid
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from app.llm import fast_llm
from app import retrieve as retrieval  # used as retrieval.bm25, so we always get the current object after a reload
from app.retrieve import search

# Best cosine similarity below this = nothing relevant, so the app abstains (tune in phase 10).
DENSE_MIN_SCORE = 0.55

SYSTEM = (
    "You answer questions using only the numbered documents below. After each claim, cite the "
    "document like [2], with plain square brackets. If the documents do not contain the answer, "
    "reply exactly: I don't know. "
    "The earlier conversation is only for understanding follow-up questions; never use it as a source."
)


answer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Earlier conversation:\n{history}\n\nRetrieved documents:\n{retrieved_docs_text}\n\nQuestion: {question}"),
])
answer_chain = answer_prompt | fast_llm | StrOutputParser()

condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the current question as one standalone search question, using the conversation "
               "to replace words like 'it' or 'that'. Reply with the question only."),
    ("human", "Conversation:\n{history}\n\nCurrent question: {question}"),
])
condense_chain = condense_prompt | fast_llm | StrOutputParser()


class State(TypedDict, total=False):
    question: str
    mode: str
    history: list[dict]   # remembered turns: {"question", "answer"}; kept by the checkpointer
    search_query: str     # the question rewritten so it makes sense alone
    retrieved_docs: list[dict]
    relevant: bool
    answer: str
    sources: list[dict]


def history_text(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for turn in history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def condense(state: State) -> dict:
    history = state.get("history", [])
    if not history:  # first question: nothing to resolve, skip the LLM call
        return {"search_query": state["question"]}
    rewritten = condense_chain.invoke({"history": history_text(history), "question": state["question"]})
    return {"search_query": rewritten.strip()}


def retrieve(state: State) -> dict:
    query = state["search_query"]
    mode = state["mode"]
    retrieved_docs = search(query, mode)

    # "relevant" decides whether we answer or say "I don't know".
    if mode == "sparse":
        # relevant if at least one word of the query appears somewhere in the documents
        scores = retrieval.bm25.vectorizer.get_scores(retrieval.bm25.preprocess_func(query))
        relevant = bool(max(scores) > 0)
    elif mode == "dense":
        # relevant if the best match is similar enough
        relevant = retrieved_docs[0]["score"] >= DENSE_MIN_SCORE
    else:
        # hybrid results have no score, so we cannot judge them. We only check that something was found.
        relevant = len(retrieved_docs) > 0

    return {"retrieved_docs": retrieved_docs, "relevant": relevant}


def route_after_retrieve(state: State) -> str:
    """Returns the name of the next node: answer if the search found relevant text, else abstain."""
    if state["relevant"]:
        return "generate"
    return "abstain"


def abstain(_state: State) -> dict:
    # LangGraph always passes the state to a node. This node does not need it
    # (the answer is fixed), so the name starts with _ to show it is unused.
    return {"answer": "I don't know.", "sources": []}


def generate(state: State) -> dict:
    retrieved_docs_text = "\n\n".join(
        f"[{doc_number}] ({retrieved_doc['file']}, page {retrieved_doc['page']}) {retrieved_doc['text']}"
        for doc_number, retrieved_doc in enumerate(state["retrieved_docs"], start=1)
    )
    answer = answer_chain.invoke({"history": history_text(state.get("history", [])),
                                  "retrieved_docs_text": retrieved_docs_text, "question": state["question"]})
    return {"answer": answer}


def cite(state: State) -> dict:
    """Sources are the retrieved documents whose number, like [2] or [1][2] etc., appears in the answer. 
    Actually the answer also contains the retrieved document's doc_number(check in "generate" function "retrieved_docs_text"), 
    so sources are the retrieved documents whose number appears in the answer. And hence these are the relevant part of the retrieved documents which is used in answer."""
    sources = []
    for doc_number, retrieved_doc in enumerate(state["retrieved_docs"], start=1):
        if f"[{doc_number}]" in state["answer"]:
            sources.append(retrieved_doc)
    return {"sources": sources}


def remember(state: State) -> dict:
    old_history = state.get("history", [])
    new_turn = {"question": state["question"], "answer": state["answer"]}
    history = old_history + [new_turn]
    # history[-3:] means "the last 3 items". It drops the oldest turns so the history cannot grow forever.
    history = history[-3:]
    return {"history": history}


graph_builder = StateGraph(State)
for node_name, node_function in [("condense", condense), ("retrieve", retrieve), ("abstain", abstain),
                                 ("generate", generate), ("cite", cite), ("remember", remember)]:
    graph_builder.add_node(node_name, node_function)
graph_builder.add_edge(START, "condense")
graph_builder.add_edge("condense", "retrieve")
graph_builder.add_conditional_edges("retrieve", route_after_retrieve)
graph_builder.add_edge("generate", "cite")
graph_builder.add_edge("cite", "remember")
graph_builder.add_edge("abstain", "remember")
graph_builder.add_edge("remember", END)

# The checkpointer saves the state per session_id, so history survives between calls.
# It lives in memory: restarting the backend forgets all sessions.
app_graph = graph_builder.compile(checkpointer=InMemorySaver())


def ask(question: str, mode: str, session_id: str | None = None, tags: list[str] = []) -> dict:
    """Without a session_id every call is a fresh chat. With one, earlier turns are remembered.
    tags are extra LangSmith tags from the caller, for example "source:ui" or "source:eval"."""
    run_id = uuid.uuid4()  # the id of this trace in LangSmith (used for the trace link and feedback)

    mode_tags = f"mode:{mode}"
    all_tags = [mode_tags] + (tags)

    config = {
        "configurable": {"thread_id": session_id or f"no-session-{run_id}"},  # memory: new chat if no session
        "run_id": run_id,                                                     # tracing
        "tags": all_tags,
    }
    final_state = app_graph.invoke({"question": question, "mode": mode}, config)
    answer = final_state["answer"]
    return {"answer": answer, "sources": final_state["sources"],
            "run_id": str(run_id),
            "abstained": answer.strip().startswith("I don't know"),  # covers both the abstain node and the model
            "retrieved_docs": final_state["retrieved_docs"],
            "relevant": final_state["relevant"],
            "search_query": final_state["search_query"]}
