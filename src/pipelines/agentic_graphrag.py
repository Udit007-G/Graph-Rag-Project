from typing import Dict
from src.agents.orchestrator import run_investigation


def run_agentic_graphrag(question: str, tg_client) -> Dict:
    """Agentic GraphRAG: orchestrated multi-step investigation."""
    result = run_investigation(question, tg_client, is_agentic=True)
    result["pipeline"] = "Agentic GraphRAG"
    return result
