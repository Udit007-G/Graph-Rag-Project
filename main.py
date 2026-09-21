import sys
import json
import argparse
from src.tg.adapter import get_client
from src.pipelines.llm_only import run_llm_only
from src.pipelines.rag import run_rag
from src.pipelines.graphrag import run_graphrag
from src.pipelines.agentic_graphrag import run_agentic_graphrag
from src.utils.benchmark import (
    load_questions, run_benchmark, print_summary, compare_all_pipelines,
    generate_html_dashboard, generate_json_dashboard
)
from config.settings import CORPUS_PATH, EVAL_PUBLIC_PATH


def run_full_benchmark(client):
    """Run all four pipelines and compare."""
    questions = load_questions(EVAL_PUBLIC_PATH)
    print(f"Loaded {len(questions)} questions")

    print("\n--- Running LLM-Only Pipeline ---")
    llm_results = run_benchmark(
        questions, run_llm_only, client,
        output_path="results/llm_only_results.json"
    )
    print_summary(llm_results, "LLM-Only")

    print("\n--- Running RAG Pipeline ---")
    rag_results = run_benchmark(
        questions, run_rag, client,
        output_path="results/rag_results.json"
    )
    print_summary(rag_results, "RAG")

    print("\n--- Running GraphRAG Pipeline ---")
    graphrag_results = run_benchmark(
        questions, run_graphrag, client,
        output_path="results/graphrag_results.json"
    )
    print_summary(graphrag_results, "GraphRAG")

    print("\n--- Running Agentic GraphRAG Pipeline ---")
    agentic_results = run_benchmark(
        questions, run_agentic_graphrag, client,
        output_path="results/agentic_results.json"
    )
    print_summary(agentic_results, "Agentic GraphRAG")

    compare_all_pipelines(llm_results, rag_results, graphrag_results, agentic_results)

    generate_html_dashboard(llm_results, rag_results, graphrag_results, agentic_results)
    generate_json_dashboard(llm_results, rag_results, graphrag_results, agentic_results)

    with open("results/comparison.json", "w") as f:
        json.dump({
            "llm_only": llm_results,
            "rag": rag_results,
            "graphrag": graphrag_results,
            "agentic": agentic_results
        }, f, indent=2)
    print("\nFull comparison saved to results/comparison.json")


def run_single_question(client, question: str, pipeline: str = "agentic"):
    """Run a single question through a specified pipeline."""
    pipelines = {
        "llm": run_llm_only,
        "rag": run_rag,
        "graphrag": run_graphrag,
        "agentic": run_agentic_graphrag
    }

    fn = pipelines.get(pipeline, run_agentic_graphrag)
    result = fn(question, client)

    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Tokens: {result.get('total_tokens', 'N/A')}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Agentic GraphRAG Hackathon")
    parser.add_argument("--mode", choices=["benchmark", "single"],
                        default="benchmark", help="Run mode")
    parser.add_argument("--pipeline", choices=["llm", "rag", "graphrag", "agentic"],
                        default="agentic", help="Pipeline for single mode")
    parser.add_argument("--question", type=str, help="Question for single mode")
    args = parser.parse_args()

    client = get_client()

    if args.mode == "benchmark":
        run_full_benchmark(client)
    elif args.mode == "single":
        if not args.question:
            print("Error: --question required for single mode")
            sys.exit(1)
        run_single_question(client, args.question, args.pipeline)


if __name__ == "__main__":
    main()
