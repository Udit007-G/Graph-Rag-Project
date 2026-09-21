import json
from typing import List, Dict
from config.settings import get_openai_client, get_model_name, llm_completion


def evaluate_evidence_sufficiency(question: str, evidence_texts: List[str]) -> Dict:
    """Evaluate whether collected evidence is sufficient to answer the question."""
    model = get_model_name()
    combined = "\n\n".join([f"[Evidence {i+1}]: {e[:500]}" for i, e in enumerate(evidence_texts)])

    prompt = f"""Evaluate if the following evidence is sufficient to answer the question.
Return a JSON object with:
- "sufficient": boolean
- "confidence": float (0-1)
- "gaps": list of strings describing what is missing
- "reasoning": brief explanation

Question: {question}

Evidence:
{combined}

Evaluation:"""

    try:
        response = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception:
        return {"sufficient": False, "confidence": 0.0, "gaps": ["Evaluation failed"], "reasoning": "Error"}


def suggest_next_action(question: str, evidence_texts: List[str], current_plan: List[str]) -> Dict:
    """Suggest the next best action for the orchestrator."""
    model = get_model_name()
    combined = "\n\n".join([f"[{i+1}]: {e[:300]}" for i, e in enumerate(evidence_texts)])

    prompt = f"""You are an investigation orchestrator. Based on the question and evidence so far,
suggest the SINGLE best next action.

Question: {question}

Current evidence:
{combined}

Actions available:
1. entity_link - Extract and link entities from the question
2. graph_traverse - Traverse the graph from known entities
3. similarity_search - Search for similar documents
4. document_retrieve - Get full text of specific documents
5. aggregate - Count/combine/filter existing evidence
6. multi_hop_reason - Chain multiple traversals
7. synthesize_answer - Form final answer (only if evidence is sufficient)

Return JSON with:
- "action": one of the action names above
- "parameters": dict of parameters for the action
- "reasoning": why this action is best
- "is_final": boolean (true if we should synthesize the answer now)
"""

    try:
        response = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception:
        return {
            "action": "synthesize_answer",
            "parameters": {},
            "reasoning": "Fallback due to evaluation error",
            "is_final": True
        }
