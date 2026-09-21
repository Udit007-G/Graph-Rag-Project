from typing import List, Dict, Any
from config.settings import get_openai_client, get_model_name, llm_completion


def aggregate_results(evidence_texts: List[str], question: str) -> str:
    """Aggregate multiple evidence pieces into a coherent summary."""
    if not evidence_texts:
        return "No evidence found."

    model = get_model_name()
    combined = "\n\n---\n\n".join(evidence_texts)

    prompt = f"""Given the following evidence pieces, provide a concise summary
that addresses the question. Only use information from the evidence.

Question: {question}

Evidence:
{combined}

Summary:"""

    response = llm_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()


def count_with_criteria(items: List[Dict], criteria: Dict[str, Any]) -> int:
    """Count items matching given criteria."""
    count = 0
    for item in items:
        match = True
        for key, value in criteria.items():
            if key in item:
                if isinstance(value, (int, float)):
                    try:
                        if float(item[key]) <= value:
                            match = False
                            break
                    except (ValueError, TypeError):
                        match = False
                        break
                elif item[key] != value:
                    match = False
                    break
            else:
                match = False
                break
        if match:
            count += 1
    return count


def aggregate_medal_counts(medalists: List[Dict], filters: Dict = None) -> Dict[str, int]:
    """Count medals by type, optionally filtered."""
    counts = {"Gold": 0, "Silver": 0, "Bronze": 0, "total": 0}

    for m in medalists:
        if filters:
            skip = False
            for k, v in filters.items():
                if k in m and m[k] != v:
                    skip = True
                    break
            if skip:
                continue

        medal = m.get("medal", "")
        if medal in counts:
            counts[medal] += 1
            counts["total"] += 1

    return counts
