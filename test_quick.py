"""Quick test of agentic pipeline on 2 questions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tg.adapter import get_client
from src.pipelines.agentic_graphrag import run_agentic_graphrag

client = get_client()

# Test 1: Aggregation
r1 = run_agentic_graphrag(
    "How many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
    client,
)
print("=== Aggregation ===")
print("Answer:", r1["answer"])
print("Tokens:", r1["tokens"])

# Test 2: Temporal
r2 = run_agentic_graphrag(
    "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?",
    client,
)
print()
print("=== Temporal ===")
print("Answer:", r2["answer"])
print("Tokens:", r2["tokens"])
