"""LangSmith dataset evaluation. Complements LangFuse online tracing with offline,
versioned dataset runs: upload a labelled {question, answer} set once, then run the
agent against it with custom evaluators (faithfulness, answer correctness) on every
change to catch retrieval / generation regressions before they ship."""
from __future__ import annotations
import os
from typing import Callable, Dict, List

from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate

from src.agents.rag_agent import build_rag_agent, run_query
from src.evaluation.langfuse_eval import score_faithfulness, score_relevance
from src.vectorstore.store import HybridStore


def get_client() -> Client:
    return Client(api_key=os.environ.get("LANGCHAIN_API_KEY"))


def create_dataset(name: str, examples: List[Dict]) -> str:
    """Create (or reuse) a LangSmith dataset from {question, answer} examples."""
    client = get_client()
    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
    else:
        ds = client.create_dataset(dataset_name=name)
    client.create_examples(
        inputs=[{"question": e["question"]} for e in examples],
        outputs=[{"answer": e.get("answer", "")} for e in examples],
        dataset_id=ds.id,
    )
    return str(ds.id)


def _faithfulness_evaluator(run, example) -> Dict:
    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")
    return {"key": "faithfulness", "score": score_faithfulness(context, answer)}


def _relevance_evaluator(run, example) -> Dict:
    answer = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")
    return {"key": "answer_relevance", "score": score_relevance(question, answer)}


def make_target(store: HybridStore) -> Callable[[Dict], Dict]:
    """Build a LangSmith target function that runs the Babel agent end-to-end."""
    agent = build_rag_agent(store)

    def target(inputs: Dict) -> Dict:
        answer = run_query(agent, inputs["question"])
        return {
            "answer": answer.with_inline_markers(),
            "context": "\n".join(d["snippet"] for d in answer.documents),
        }

    return target


def run_evaluation(store: HybridStore, dataset_name: str, experiment_prefix: str = "babel"):
    """Run the agent over a LangSmith dataset with the faithfulness + relevance evaluators."""
    return ls_evaluate(
        make_target(store),
        data=dataset_name,
        evaluators=[_faithfulness_evaluator, _relevance_evaluator],
        experiment_prefix=experiment_prefix,
        client=get_client(),
    )
