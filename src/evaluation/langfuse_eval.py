"""LangFuse tracing + RAG scoring (retrieval recall, faithfulness, answer
relevance). Traces every agent run and attaches numeric scores so retrieval and
generation quality can be tracked over time and regressions caught in CI."""
from __future__ import annotations
import os
from typing import Dict, List, Optional

import cohere
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

from src.agents.rag_agent import run_query
from src.clients import get_client

_langfuse: Optional[Langfuse] = None


def init_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _langfuse


@observe(name="babel_rag_query")
def traced_query(agent, question: str) -> str:
    answer = run_query(agent, question)
    rendered = answer.with_inline_markers()
    langfuse_context.update_current_trace(input=question, output=rendered)
    return rendered


def retrieval_recall(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """Fraction of gold document ids present in the retrieved set."""
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids) & set(gold_ids)) / len(set(gold_ids))


def score_faithfulness(context: str, answer: str,
                       client: Optional[cohere.Client] = None) -> float:
    """LLM-judge faithfulness in [0, 1]: is the answer supported by the context?"""
    client = client or get_client()
    resp = client.chat(
        model="command-r",
        preamble="Score 0-1 how fully the ANSWER is supported by the CONTEXT. "
                 "Reply with only a float.",
        message=f"Context:\n{context}\n\nAnswer:\n{answer}",
        temperature=0.0,
    )
    return _parse_float(resp.text)


def score_relevance(question: str, answer: str,
                    client: Optional[cohere.Client] = None) -> float:
    client = client or get_client()
    resp = client.chat(
        model="command-r",
        preamble="Score 0-1 how well the ANSWER addresses the QUESTION. "
                 "Reply with only a float.",
        message=f"Question: {question}\nAnswer: {answer}",
        temperature=0.0,
    )
    return _parse_float(resp.text)


def evaluate_batch(qa_pairs: List[Dict], agent, lf: Optional[Langfuse] = None) -> List[Dict]:
    lf = lf or init_langfuse()
    results = []
    for pair in qa_pairs:
        q = pair["question"]
        answer = run_query(agent, q)
        context = "\n".join(d["snippet"] for d in answer.documents)
        faith = score_faithfulness(context, answer.text)
        rel = score_relevance(q, answer.text)
        results.append({"question": q, "faithfulness": faith, "relevance": rel})
        lf.score(name="faithfulness", value=faith, comment=q)
        lf.score(name="relevance", value=rel, comment=q)
    return results


def _parse_float(text: str) -> float:
    try:
        return min(1.0, max(0.0, float(text.strip().split()[0])))
    except (ValueError, IndexError):
        return 0.0
