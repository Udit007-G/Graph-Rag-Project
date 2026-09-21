import json
import os
from typing import List, Dict
from datetime import datetime


def load_questions(path: str) -> List[Dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    return answer.lower().strip().rstrip(".").replace(",", "")


def check_answer_match(predicted: str, gold_answers: List[str]) -> bool:
    """Check if predicted answer matches any gold answer."""
    norm_pred = normalize_answer(predicted)
    for gold in gold_answers:
        if normalize_answer(gold) in norm_pred or norm_pred in normalize_answer(gold):
            return True
    return False


def evaluate_single(predicted: str, gold_answers: List[str], qtype: str) -> Dict:
    """Evaluate a single question's answer."""
    match = check_answer_match(predicted, gold_answers)

    return {
        "correct": match,
        "predicted": predicted,
        "gold": gold_answers,
        "qtype": qtype
    }


def run_benchmark(questions: List[Dict], pipeline_fn, tg_client, output_path: str = None) -> Dict:
    """Run benchmark evaluation across all questions."""
    results = []
    total_tokens = 0
    correct_by_type = {}
    total_by_type = {}

    for i, q in enumerate(questions):
        qid = q.get("qid", f"q-{i}")
        question = q["question"]
        gold_answers = q.get("answer", [])
        qtype = q.get("qtype", "unknown")

        print(f"[{i+1}/{len(questions)}] {qid}: {question[:60]}...")

        try:
            result = pipeline_fn(question, tg_client)
            predicted = result.get("answer", "")
            tokens = result.get("total_tokens", 0)
        except Exception as e:
            print(f"  ERROR: {e}")
            predicted = ""
            tokens = 0
            result = {"error": str(e)}

        eval_result = evaluate_single(predicted, gold_answers, qtype)
        eval_result["qid"] = qid
        eval_result["tokens"] = tokens
        results.append(eval_result)

        total_tokens += tokens

        if qtype not in correct_by_type:
            correct_by_type[qtype] = 0
            total_by_type[qtype] = 0
        total_by_type[qtype] += 1
        if eval_result["correct"]:
            correct_by_type[qtype] += 1

        print(f"  {'CORRECT' if eval_result['correct'] else 'WRONG'} | Tokens: {tokens}")

    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0

    type_accuracy = {}
    for qtype in total_by_type:
        type_accuracy[qtype] = correct_by_type[qtype] / total_by_type[qtype] if total_by_type[qtype] > 0 else 0

    summary = {
        "total_questions": len(questions),
        "correct": sum(1 for r in results if r["correct"]),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "avg_tokens_per_question": total_tokens / len(questions) if questions else 0,
        "accuracy_by_type": type_accuracy,
        "results": results
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return summary


def print_summary(summary: Dict, pipeline_name: str):
    """Print a formatted summary."""
    print(f"\n{'='*60}")
    print(f" {pipeline_name} Results")
    print(f"{'='*60}")
    print(f" Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total_questions']})")
    print(f" Total Tokens: {summary['total_tokens']:,}")
    print(f" Avg Tokens/Q: {summary['avg_tokens_per_question']:,.0f}")
    print(f"\n Accuracy by Type:")
    for qtype, acc in summary.get("accuracy_by_type", {}).items():
        print(f"   {qtype}: {acc:.1%}")
    print(f"{'='*60}")


def compare_pipelines(rag_summary: Dict, graphrag_summary: Dict, agentic_summary: Dict):
    """Generate comparison table across three pipelines."""
    print(f"\n{'='*80}")
    print(f" Pipeline Comparison")
    print(f"{'='*80}")
    print(f" {'Metric':<25} {'RAG':>12} {'GraphRAG':>12} {'Agentic':>12}")
    print(f" {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f" {'Accuracy':<25} {rag_summary['accuracy']:>11.1%} {graphrag_summary['accuracy']:>11.1%} {agentic_summary['accuracy']:>11.1%}")
    print(f" {'Correct':<25} {rag_summary['correct']:>12} {graphrag_summary['correct']:>12} {agentic_summary['correct']:>12}")
    print(f" {'Total Tokens':<25} {rag_summary['total_tokens']:>12,} {graphrag_summary['total_tokens']:>12,} {agentic_summary['total_tokens']:>12,}")
    print(f" {'Avg Tokens/Q':<25} {rag_summary['avg_tokens_per_question']:>12,.0f} {graphrag_summary['avg_tokens_per_question']:>12,.0f} {agentic_summary['avg_tokens_per_question']:>12,.0f}")
    print(f"{'='*80}")

    all_types = set(list(rag_summary.get("accuracy_by_type", {}).keys()) +
                    list(graphrag_summary.get("accuracy_by_type", {}).keys()) +
                    list(agentic_summary.get("accuracy_by_type", {}).keys()))

    if all_types:
        print(f"\n Accuracy by Question Type:")
        print(f" {'Type':<20} {'RAG':>12} {'GraphRAG':>12} {'Agentic':>12}")
        print(f" {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        for qtype in sorted(all_types):
            r = rag_summary.get("accuracy_by_type", {}).get(qtype, 0)
            g = graphrag_summary.get("accuracy_by_type", {}).get(qtype, 0)
            a = agentic_summary.get("accuracy_by_type", {}).get(qtype, 0)
            print(f" {qtype:<20} {r:>11.1%} {g:>11.1%} {a:>11.1%}")


def compare_all_pipelines(llm_summary: Dict, rag_summary: Dict, graphrag_summary: Dict, agentic_summary: Dict):
    """Generate comparison table across all four pipelines."""
    print(f"\n{'='*95}")
    print(f" Full Pipeline Comparison (LLM-Only / RAG / GraphRAG / Agentic)")
    print(f"{'='*95}")
    print(f" {'Metric':<25} {'LLM-Only':>12} {'RAG':>12} {'GraphRAG':>12} {'Agentic':>12}")
    print(f" {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    print(f" {'Accuracy':<25} {llm_summary['accuracy']:>11.1%} {rag_summary['accuracy']:>11.1%} {graphrag_summary['accuracy']:>11.1%} {agentic_summary['accuracy']:>11.1%}")
    print(f" {'Correct':<25} {llm_summary['correct']:>12} {rag_summary['correct']:>12} {graphrag_summary['correct']:>12} {agentic_summary['correct']:>12}")
    print(f" {'Total Tokens':<25} {llm_summary['total_tokens']:>12,} {rag_summary['total_tokens']:>12,} {graphrag_summary['total_tokens']:>12,} {agentic_summary['total_tokens']:>12,}")
    print(f" {'Avg Tokens/Q':<25} {llm_summary['avg_tokens_per_question']:>12,.0f} {rag_summary['avg_tokens_per_question']:>12,.0f} {graphrag_summary['avg_tokens_per_question']:>12,.0f} {agentic_summary['avg_tokens_per_question']:>12,.0f}")
    print(f"{'='*95}")

    all_types = set(list(llm_summary.get("accuracy_by_type", {}).keys()) +
                    list(rag_summary.get("accuracy_by_type", {}).keys()) +
                    list(graphrag_summary.get("accuracy_by_type", {}).keys()) +
                    list(agentic_summary.get("accuracy_by_type", {}).keys()))

    if all_types:
        print(f"\n Accuracy by Question Type:")
        print(f" {'Type':<20} {'LLM-Only':>12} {'RAG':>12} {'GraphRAG':>12} {'Agentic':>12}")
        print(f" {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for qtype in sorted(all_types):
            l = llm_summary.get("accuracy_by_type", {}).get(qtype, 0)
            r = rag_summary.get("accuracy_by_type", {}).get(qtype, 0)
            g = graphrag_summary.get("accuracy_by_type", {}).get(qtype, 0)
            a = agentic_summary.get("accuracy_by_type", {}).get(qtype, 0)
            print(f" {qtype:<20} {l:>11.1%} {r:>11.1%} {g:>11.1%} {a:>11.1%}")


def generate_html_dashboard(llm_summary: Dict, rag_summary: Dict, graphrag_summary: Dict, agentic_summary: Dict, output_path: str = "results/dashboard.html"):
    """Generate an HTML metrics dashboard."""
    all_types = sorted(set(
        list(llm_summary.get("accuracy_by_type", {}).keys()) +
        list(rag_summary.get("accuracy_by_type", {}).keys()) +
        list(graphrag_summary.get("accuracy_by_type", {}).keys()) +
        list(agentic_summary.get("accuracy_by_type", {}).keys())
    ))

    type_rows = ""
    for qtype in all_types:
        l = llm_summary.get("accuracy_by_type", {}).get(qtype, 0) * 100
        r = rag_summary.get("accuracy_by_type", {}).get(qtype, 0) * 100
        g = graphrag_summary.get("accuracy_by_type", {}).get(qtype, 0) * 100
        a = agentic_summary.get("accuracy_by_type", {}).get(qtype, 0) * 100
        type_rows += f"""
        <tr>
            <td>{qtype}</td>
            <td>{l:.1f}%</td>
            <td>{r:.1f}%</td>
            <td>{g:.1f}%</td>
            <td>{a:.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic GraphRAG Hackathon - Metrics Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; font-size: 2rem; margin-bottom: 2rem; color: #38bdf8; }}
        .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 2rem; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #334155; }}
        .card h3 {{ color: #94a3b8; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }}
        .card .value {{ font-size: 2.5rem; font-weight: 700; }}
        .card .sub {{ color: #64748b; font-size: 0.875rem; margin-top: 0.25rem; }}
        .llm {{ color: #f87171; }}
        .rag {{ color: #fbbf24; }}
        .graphrag {{ color: #34d399; }}
        .agentic {{ color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 12px; overflow: hidden; margin-bottom: 2rem; }}
        th {{ background: #334155; padding: 1rem; text-align: left; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }}
        td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #334155; }}
        tr:hover {{ background: #334155; }}
        .section {{ margin-bottom: 2rem; }}
        .section h2 {{ font-size: 1.25rem; margin-bottom: 1rem; color: #f1f5f9; }}
        .bar-chart {{ display: flex; align-items: end; gap: 1rem; height: 200px; padding: 1rem; background: #1e293b; border-radius: 12px; }}
        .bar-group {{ flex: 1; display: flex; flex-direction: column; align-items: center; gap: 0.5rem; }}
        .bar {{ width: 100%; border-radius: 4px 4px 0 0; min-height: 4px; transition: height 0.3s; }}
        .bar-label {{ font-size: 0.75rem; color: #94a3b8; }}
        .legend {{ display: flex; gap: 1.5rem; justify-content: center; margin-top: 1rem; }}
        .legend-item {{ display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Agentic GraphRAG Hackathon - Metrics Dashboard</h1>

        <div class="cards">
            <div class="card">
                <h3>LLM-Only</h3>
                <div class="value llm">{llm_summary['accuracy']:.1%}</div>
                <div class="sub">{llm_summary['correct']}/{llm_summary['total_questions']} correct</div>
                <div class="sub">{llm_summary['total_tokens']:,} tokens</div>
            </div>
            <div class="card">
                <h3>RAG</h3>
                <div class="value rag">{rag_summary['accuracy']:.1%}</div>
                <div class="sub">{rag_summary['correct']}/{rag_summary['total_questions']} correct</div>
                <div class="sub">{rag_summary['total_tokens']:,} tokens</div>
            </div>
            <div class="card">
                <h3>GraphRAG</h3>
                <div class="value graphrag">{graphrag_summary['accuracy']:.1%}</div>
                <div class="sub">{graphrag_summary['correct']}/{graphrag_summary['total_questions']} correct</div>
                <div class="sub">{graphrag_summary['total_tokens']:,} tokens</div>
            </div>
            <div class="card">
                <h3>Agentic GraphRAG</h3>
                <div class="value agentic">{agentic_summary['accuracy']:.1%}</div>
                <div class="sub">{agentic_summary['correct']}/{agentic_summary['total_questions']} correct</div>
                <div class="sub">{agentic_summary['total_tokens']:,} tokens</div>
            </div>
        </div>

        <div class="section">
            <h2>Accuracy by Question Type</h2>
            <table>
                <thead>
                    <tr>
                        <th>Question Type</th>
                        <th>LLM-Only</th>
                        <th>RAG</th>
                        <th>GraphRAG</th>
                        <th>Agentic GraphRAG</th>
                    </tr>
                </thead>
                <tbody>
                    {type_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Accuracy Comparison</h2>
            <div class="bar-chart">
                <div class="bar-group">
                    <div class="bar llm" style="height: {llm_summary['accuracy']*100*2}px; background: #f87171;"></div>
                    <div class="bar-label">LLM-Only</div>
                </div>
                <div class="bar-group">
                    <div class="bar rag" style="height: {rag_summary['accuracy']*100*2}px; background: #fbbf24;"></div>
                    <div class="bar-label">RAG</div>
                </div>
                <div class="bar-group">
                    <div class="bar graphrag" style="height: {graphrag_summary['accuracy']*100*2}px; background: #34d399;"></div>
                    <div class="bar-label">GraphRAG</div>
                </div>
                <div class="bar-group">
                    <div class="bar agentic" style="height: {agentic_summary['accuracy']*100*2}px; background: #38bdf8;"></div>
                    <div class="bar-label">Agentic</div>
                </div>
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background: #f87171;"></div> LLM-Only</div>
                <div class="legend-item"><div class="legend-dot" style="background: #fbbf24;"></div> RAG</div>
                <div class="legend-item"><div class="legend-dot" style="background: #34d399;"></div> GraphRAG</div>
                <div class="legend-item"><div class="legend-dot" style="background: #38bdf8;"></div> Agentic GraphRAG</div>
            </div>
        </div>

        <div class="section">
            <h2>Token Efficiency</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>LLM-Only</th>
                        <th>RAG</th>
                        <th>GraphRAG</th>
                        <th>Agentic GraphRAG</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Total Tokens</td>
                        <td>{llm_summary['total_tokens']:,}</td>
                        <td>{rag_summary['total_tokens']:,}</td>
                        <td>{graphrag_summary['total_tokens']:,}</td>
                        <td>{agentic_summary['total_tokens']:,}</td>
                    </tr>
                    <tr>
                        <td>Avg Tokens/Question</td>
                        <td>{llm_summary['avg_tokens_per_question']:,.0f}</td>
                        <td>{rag_summary['avg_tokens_per_question']:,.0f}</td>
                        <td>{graphrag_summary['avg_tokens_per_question']:,.0f}</td>
                        <td>{agentic_summary['avg_tokens_per_question']:,.0f}</td>
                    </tr>
                    <tr>
                        <td>Cost Efficiency (correct/token)</td>
                        <td>{llm_summary['correct']/max(llm_summary['total_tokens'],1)*1e6:.2f}</td>
                        <td>{rag_summary['correct']/max(rag_summary['total_tokens'],1)*1e6:.2f}</td>
                        <td>{graphrag_summary['correct']/max(graphrag_summary['total_tokens'],1)*1e6:.2f}</td>
                        <td>{agentic_summary['correct']/max(agentic_summary['total_tokens'],1)*1e6:.2f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nDashboard saved to {output_path}")


def generate_json_dashboard(llm_summary: Dict, rag_summary: Dict, graphrag_summary: Dict, agentic_summary: Dict, output_path: str = "results/dashboard.json"):
    """Generate a JSON metrics dashboard for programmatic access."""
    dashboard = {
        "pipelines": {
            "llm_only": {
                "accuracy": llm_summary["accuracy"],
                "correct": llm_summary["correct"],
                "total": llm_summary["total_questions"],
                "total_tokens": llm_summary["total_tokens"],
                "avg_tokens_per_question": llm_summary["avg_tokens_per_question"],
                "accuracy_by_type": llm_summary.get("accuracy_by_type", {})
            },
            "rag": {
                "accuracy": rag_summary["accuracy"],
                "correct": rag_summary["correct"],
                "total": rag_summary["total_questions"],
                "total_tokens": rag_summary["total_tokens"],
                "avg_tokens_per_question": rag_summary["avg_tokens_per_question"],
                "accuracy_by_type": rag_summary.get("accuracy_by_type", {})
            },
            "graphrag": {
                "accuracy": graphrag_summary["accuracy"],
                "correct": graphrag_summary["correct"],
                "total": graphrag_summary["total_questions"],
                "total_tokens": graphrag_summary["total_tokens"],
                "avg_tokens_per_question": graphrag_summary["avg_tokens_per_question"],
                "accuracy_by_type": graphrag_summary.get("accuracy_by_type", {})
            },
            "agentic": {
                "accuracy": agentic_summary["accuracy"],
                "correct": agentic_summary["correct"],
                "total": agentic_summary["total_questions"],
                "total_tokens": agentic_summary["total_tokens"],
                "avg_tokens_per_question": agentic_summary["avg_tokens_per_question"],
                "accuracy_by_type": agentic_summary.get("accuracy_by_type", {})
            }
        },
        "comparison": {
            "agentic_vs_rag_accuracy_delta": agentic_summary["accuracy"] - rag_summary["accuracy"],
            "agentic_vs_graphrag_accuracy_delta": agentic_summary["accuracy"] - graphrag_summary["accuracy"],
            "agentic_token_overhead": agentic_summary["total_tokens"] / max(rag_summary["total_tokens"], 1),
        }
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"JSON dashboard saved to {output_path}")
