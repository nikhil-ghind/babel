"""Agentic RAG with a self-correction loop, built on LangGraph.

Flow:  retrieve → generate → grade → (rewrite → retrieve …) | END

The grade node asks Cohere Command to judge whether the grounded answer actually
resolves the question from the retrieved passages. If it is insufficient and we
have retries left, the query is rewritten and the loop runs again with a wider
net; otherwise the best answer so far is returned. This is what turns a one-shot
RAG call into a self-correcting agent."""
from __future__ import annotations
from typing import List, Optional, TypedDict

import cohere
from langgraph.graph import END, START, StateGraph

from src.clients import get_client
from src.generation.grounded import GroundedAnswer, generate
from src.retrieval.hybrid import retrieve
from src.retrieval.query_expansion import rewrite_query
from src.retrieval.rerank import Ranked
from src.vectorstore.store import HybridStore

_GRADE_PREAMBLE = (
    "You are a strict grader. Given a question, retrieved passages, and a candidate "
    "answer, reply with exactly 'PASS' if the answer is fully grounded in the passages "
    "and resolves the question, otherwise 'FAIL'."
)


class AgentState(TypedDict, total=False):
    query: str            # current (possibly rewritten) query
    original_query: str
    passages: List[Ranked]
    answer: GroundedAnswer
    iterations: int
    verdict: str


def _grade(query: str, answer: GroundedAnswer, client: cohere.Client) -> str:
    context = "\n".join(d["snippet"] for d in answer.documents)
    resp = client.chat(
        model="command-r",
        preamble=_GRADE_PREAMBLE,
        message=f"Question: {query}\n\nPassages:\n{context}\n\nAnswer:\n{answer.text}",
        temperature=0.0,
    )
    return "PASS" if "PASS" in resp.text.upper() else "FAIL"


def build_rag_agent(store: HybridStore, max_iterations: int = 3,
                    top_n: int = 5, client: Optional[cohere.Client] = None):
    client = client or get_client()

    def retrieve_node(state: AgentState) -> dict:
        passages = retrieve(store, state["query"], top_n=top_n, client=client)
        return {"passages": passages, "iterations": state.get("iterations", 0) + 1}

    def generate_node(state: AgentState) -> dict:
        return {"answer": generate(state["original_query"], state["passages"], client=client)}

    def grade_node(state: AgentState) -> dict:
        return {"verdict": _grade(state["original_query"], state["answer"], client)}

    def rewrite_node(state: AgentState) -> dict:
        refined = rewrite_query(
            f"{state['original_query']} (previous answer was insufficient; "
            f"broaden or rephrase)", client=client)
        return {"query": refined}

    def route(state: AgentState) -> str:
        if state["verdict"] == "PASS" or state["iterations"] >= max_iterations:
            return "end"
        return "retry"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges("grade", route, {"retry": "rewrite", "end": END})
    graph.add_edge("rewrite", "retrieve")

    return graph.compile()


def run_query(agent, question: str) -> GroundedAnswer:
    state = agent.invoke({
        "query": question,
        "original_query": question,
        "iterations": 0,
    })
    return state["answer"]
