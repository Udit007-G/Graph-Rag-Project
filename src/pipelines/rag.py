"""RAG pipeline: keyword search + structured extraction + LLM answer."""
import re
from typing import Dict, List
from config.settings import get_openai_client, get_model_name, llm_completion
from src.agents.similarity_search import hybrid_search
from src.utils.helpers import (
    classify_question, resolve_temporal_reference, extract_event_details,
    extract_gold_from_text, extract_competitor_count, extract_nation_count,
)


def run_rag(question: str, tg_client) -> Dict:
    """RAG: retrieve relevant docs, extract structured info, generate answer."""
    q_type = classify_question(question)
    details = extract_event_details(question)
    year = details.get("year", 0)
    season = details.get("season", "")
    sport = details.get("sport", "")

    if q_type == "temporal" and year:
        search_query = f"{sport} at the {year} {season} Olympics".strip()
    elif sport and year:
        search_query = f"{sport} at the {year} {season} Olympics"
    else:
        search_query = question

    print(f"  [RAG] Search query: {search_query}")

    results = hybrid_search(search_query, top_k=15, client=tg_client)

    context_parts = []
    structured_data = []
    for r in results:
        text = r.get("text", "")[:2000]
        title = r.get("title", r.get("doc_id", ""))
        if text:
            info = {"title": title, "doc_id": r.get("doc_id", "")}

            comp = extract_competitor_count(text)
            if comp:
                info["competitor_count"] = comp

            nations = extract_nation_count(text)
            if nations:
                info["nation_count"] = nations

            gold = extract_gold_from_text(text)
            if gold:
                info["gold"] = gold

            noc_match = re.search(r"goldNOC:\s*(\w+)", text, re.IGNORECASE)
            if noc_match:
                info["goldNOC"] = noc_match.group(1)

            structured_data.append(info)

            infobox_end = text.find("\n\n")
            infobox = text[:infobox_end] if infobox_end > 0 else text[:600]
            context_parts.append(f"[{title}]\n{infobox}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

    if structured_data:
        summary_lines = ["\n=== STRUCTURED DATA ==="]
        for sd in structured_data:
            parts = []
            if "title" in sd:
                parts.append(f"Event: {sd['title']}")
            if "competitor_count" in sd:
                parts.append(f"Competitors: {sd['competitor_count']}")
            if "nation_count" in sd:
                parts.append(f"Nations: {sd['nation_count']}")
            if "gold" in sd:
                parts.append(f"Gold: {sd['gold']}")
            if "goldNOC" in sd:
                parts.append(f"GoldNOC: {sd['goldNOC']}")
            summary_lines.append(" | ".join(parts))

        if q_type == "aggregation":
            threshold_match = re.search(r"more than (\d+)", question)
            if threshold_match:
                threshold = int(threshold_match.group(1))
                count_gt = sum(
                    1 for sd in structured_data
                    if sd.get("competitor_count", 0) > threshold
                )
                summary_lines.append(
                    f"\nCOUNT of events with >{threshold} competitors: {count_gt}"
                )

        if q_type == "lookup":
            for sd in structured_data:
                if "nation_count" in sd:
                    summary_lines.append(f"\nNations in {sd.get('title', 'event')}: {sd['nation_count']}")

        context += "\n\n" + "\n".join(summary_lines)

    type_instructions = ""
    if q_type == "temporal":
        type_instructions = "\nThe previous Olympics has been resolved. Find the gold medalist for the specific event."
    elif q_type == "aggregation":
        type_instructions = "\nFor counting: count ALL events where competitors > threshold. Give ONLY the final count number."
    elif q_type == "lookup":
        type_instructions = "\nFor 'how many nations': find the specific event and give the nations count."
    elif q_type == "superlative":
        type_instructions = "\nFor superlative: identify the SINGLE event with the most competitors."
    elif q_type == "venue_date":
        type_instructions = "\nFor venue+date: find the event at the specific venue on the given date."

    prompt = f"""Answer this question using ONLY the provided context and structured data.
Be precise and cite facts from the context.
Look at the STRUCTURED DATA section for competitor counts, medalists, nations, etc.{type_instructions}

Question: {question}

Context:
{context}

Answer:"""

    try:
        response = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
        tokens = response.usage.prompt_tokens + response.usage.completion_tokens
    except Exception as e:
        answer = f"Error: {e}"
        tokens = 0

    return {
        "pipeline": "RAG",
        "question": question,
        "answer": answer,
        "docs_retrieved": len(results),
        "total_tokens": tokens,
    }
