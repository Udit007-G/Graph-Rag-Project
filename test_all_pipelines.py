"""Test all pipelines on sample questions from each question type."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from src.tg.adapter import get_client
from src.pipelines.llm_only import run_llm_only
from src.pipelines.rag import run_rag
from src.pipelines.graphrag import run_graphrag
from src.pipelines.agentic_graphrag import run_agentic_graphrag


TEST_QUESTIONS = [
    {
        "qid": "test-temporal",
        "question": "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?",
        "qtype": "temporal",
        "answer": ["Chen Ding"]
    },
    {
        "qid": "test-aggregation",
        "question": "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
        "qtype": "aggregation",
        "answer": ["5"]
    },
    {
        "qid": "test-lookup",
        "question": "How many nations competed in Sailing at the 2016 Summer Olympics – Women's RS:X?",
        "qtype": "lookup",
        "answer": ["26"]
    },
    {
        "qid": "test-superlative",
        "question": "According to the provided corpus, which shooting event at the 2008 Summer Olympics had the highest number of competitors?",
        "qtype": "superlative",
        "answer": ["Shooting at the 2008 Summer Olympics – Men's 50 metre rifle prone"]
    },
    {
        "qid": "test-venue",
        "question": "Who won the gold medal in the event held at Olympic Weightlifting Gymnasium on 20 September 1988?",
        "qtype": "multi_hop",
        "answer": ["Naim Süleymanoğlu"]
    },
]


def check_answer(predicted: str, gold_answers: list) -> bool:
    """Check if predicted answer matches any gold answer."""
    pred_lower = predicted.lower().strip()
    for gold in gold_answers:
        gold_lower = gold.lower().strip()
        if gold_lower in pred_lower or pred_lower in gold_lower:
            return True
    return False


def main():
    client = get_client()
    print(f"Client type: {type(client).__name__}")
    print("=" * 80)

    results = {
        "rag": {"correct": 0, "total": 0, "tokens": 0},
        "graphrag": {"correct": 0, "total": 0, "tokens": 0},
        "agentic": {"correct": 0, "total": 0, "tokens": 0},
    }

    for q in TEST_QUESTIONS:
        qid = q["qid"]
        question = q["question"]
        gold = q["answer"]
        qtype = q["qtype"]

        print(f"\n{'='*80}")
        print(f"Q: {qid} ({qtype})")
        print(f"Q: {question}")
        print(f"Gold: {gold}")

        # RAG
        try:
            rag = run_rag(question, client)
            rag_correct = check_answer(rag["answer"], gold)
            results["rag"]["correct"] += rag_correct
            results["rag"]["total"] += 1
            results["rag"]["tokens"] += rag.get("total_tokens", 0)
            print(f"\n  RAG:      {'CORRECT' if rag_correct else 'WRONG'} | {rag['answer'][:100]}")
            print(f"           Tokens: {rag.get('total_tokens', 0)}")
        except Exception as e:
            print(f"\n  RAG:      ERROR: {e}")

        # GraphRAG
        try:
            gr = run_graphrag(question, client)
            gr_correct = check_answer(gr["answer"], gold)
            results["graphrag"]["correct"] += gr_correct
            results["graphrag"]["total"] += 1
            results["graphrag"]["tokens"] += gr.get("total_tokens", 0)
            print(f"  GraphRAG: {'CORRECT' if gr_correct else 'WRONG'} | {gr['answer'][:100]}")
            print(f"           Tokens: {gr.get('total_tokens', 0)}")
        except Exception as e:
            print(f"  GraphRAG: ERROR: {e}")

        # Agentic
        try:
            ag = run_agentic_graphrag(question, client)
            ag_correct = check_answer(ag["answer"], gold)
            results["agentic"]["correct"] += ag_correct
            results["agentic"]["total"] += 1
            results["agentic"]["tokens"] += ag.get("tokens", 0)
            print(f"  Agentic:  {'CORRECT' if ag_correct else 'WRONG'} | {ag['answer'][:100]}")
            print(f"           Tokens: {ag.get('tokens', 0)} | Steps: {ag.get('steps', 0)}")
        except Exception as e:
            print(f"  Agentic:  ERROR: {e}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for pipe, data in results.items():
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"  {pipe.upper():12} {data['correct']}/{data['total']} ({acc:.0f}%) | {data['tokens']:,} tokens")


if __name__ == "__main__":
    main()
