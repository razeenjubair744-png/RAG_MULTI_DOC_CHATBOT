"""
Evaluate the RAG pipeline using RAGAS metrics.

Metrics measured:
  - faithfulness     : Is the answer grounded in the retrieved context?
  - answer_relevancy : Is the answer relevant to the question?
  - context_recall   : Did we retrieve the chunks needed to answer?
  - context_precision: Are the retrieved chunks actually useful?

Usage:
    python eval/run_eval.py

Output:
    eval/results.json  — per-question scores
    eval/summary.json  — mean scores (put these in your README)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.chains.rag_chain import build_retriever

EVAL_DIR = Path(__file__).parent
TEST_QUERIES_FILE = EVAL_DIR / "test_queries.json"
RESULTS_FILE = EVAL_DIR / "results.json"
SUMMARY_FILE = EVAL_DIR / "summary.json"


def load_test_queries() -> List[Dict]:
    """
    Load test Q&A pairs from JSON.
    Format: [{"question": "...", "ground_truth": "..."}]
    """
    with open(TEST_QUERIES_FILE) as f:
        return json.load(f)


def run_rag_for_eval(queries: List[Dict]) -> Dict[str, List]:
    """Run each query through the retriever + LLM and collect outputs."""
    from langchain_openai import ChatOpenAI
    from app.chains.rag_chain import ask

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {item['question'][:60]}...")
        result = ask(item["question"], session_id=f"eval_{i}")

        questions.append(item["question"])
        answers.append(result["answer"])
        ground_truths.append(item["ground_truth"])

        # Collect context snippets used
        contexts.append([s["snippet"] for s in result["sources"]])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def main():
    print("\n" + "="*50)
    print("RAGAS Evaluation")
    print("="*50)

    # 1. Load test queries
    if not TEST_QUERIES_FILE.exists():
        print(f"ERROR: {TEST_QUERIES_FILE} not found.")
        print("Create it using the template in eval/test_queries.json.example")
        return

    queries = load_test_queries()
    print(f"Loaded {len(queries)} test queries\n")

    # 2. Run RAG for each query
    print("Running RAG pipeline on test queries...")
    data = run_rag_for_eval(queries)

    # 3. Evaluate with RAGAS
    print("\nRunning RAGAS evaluation...")
    dataset = Dataset.from_dict(data)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    # 4. Save and display results
    df = result.to_pandas()

    summary = {
        "faithfulness": round(float(df["faithfulness"].mean()), 3),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 3),
        "context_recall": round(float(df["context_recall"].mean()), 3),
        "context_precision": round(float(df["context_precision"].mean()), 3),
        "num_questions": len(queries),
    }

    with open(RESULTS_FILE, "w") as f:
        f.write(df.to_json(orient="records", indent=2))

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print("RAGAS Results Summary")
    print("="*50)
    for metric, score in summary.items():
        if metric != "num_questions":
            bar = "█" * int(score * 20)
            print(f"  {metric:<22} {score:.3f}  {bar}")
    print(f"\n  Questions evaluated: {summary['num_questions']}")
    print(f"\nDetailed results → {RESULTS_FILE}")
    print(f"Summary          → {SUMMARY_FILE}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
