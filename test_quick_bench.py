import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from src.tg.adapter import get_client
from src.pipelines.llm_only import run_llm_only
from src.pipelines.rag import run_rag
from src.pipelines.graphrag import run_graphrag
from src.pipelines.agentic_graphrag import run_agentic_graphrag
from src.utils.benchmark import check_answer_match

client = get_client()
questions = [
    ("How many biathlon events at the 2018 Winter Olympics had more than 73 competitors?", ["5"]),
    ("Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?", ["Chen Ding"]),
    ("How many shooting events at the 2004 Summer Olympics had more than 30 competitors?", ["8"]),
]

for q, expected in questions:
    print("\nQ:", q[:60])
    for name, fn in [("LLM", run_llm_only), ("RAG", run_rag), ("GraphRAG", run_graphrag), ("Agentic", run_agentic_graphrag)]:
        t0 = time.time()
        r = fn(q, client)
        match = check_answer_match(r["answer"], expected)
        print("  %s: %s (%.1fs)" % ("OK" if match else "WRONG", name, time.time()-t0))
