from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from app.llm import fast_llm
from app.retrieve import bm25, search

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


def is_relevant(query: str, mode: str, retrieved_docs: list[dict]) -> bool:
    if not retrieved_docs:
        return False
    if mode == "sparse":
        return bool(max(bm25.vectorizer.get_scores(bm25.preprocess_func(query))) > 0)
    # dense and hybrid: use the best dense cosine score (hybrid results have no score of their own)
    best_doc = retrieved_docs[0] if mode == "dense" else search(query, "dense", k=1)[0]
    return bool(best_doc["score"] >= DENSE_MIN_SCORE)


def retrieve(state: State) -> dict:
    retrieved_docs = search(state["search_query"], state["mode"])
    relevant = is_relevant(state["search_query"], state["mode"], retrieved_docs)
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
    """Sources are the retrieved documents whose number, like [2], appears in the answer."""
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


def ask(question: str, mode: str, session_id: str | None = None) -> dict:
    """Without a session_id every call is a fresh chat. With one, earlier turns are remembered."""
    config = {"configurable": {"thread_id": session_id or "no-session-" + question}}
    final_state = app_graph.invoke({"question": question, "mode": mode}, config)
    answer = final_state["answer"]
    return {"answer": answer, "sources": final_state["sources"],
            "abstained": answer.strip().startswith("I don't know"),  # covers both the abstain node and the model
            "retrieved_docs": final_state["retrieved_docs"],
            "relevant": final_state["relevant"],
            "search_query": final_state["search_query"]}
