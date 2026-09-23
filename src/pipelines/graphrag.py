"""GraphRAG pipeline: official GraphRAG-style retrieval + LLM answer."""
import re
from typing import Dict
from config.settings import llm_completion
from src.graphrag_retriever import GraphRAGRetriever, HybridRetriever, CommunityRetriever, EntityRelationshipRetriever
from src.utils.helpers import (
    classify_question, resolve_temporal_reference, extract_event_details,
    extract_gold_from_text, extract_competitor_count,
)


def run_graphrag(question: str, tg_client) -> Dict:
    """GraphRAG: hybrid retrieval + graph context + LLM answer."""
    total_tokens = 0
    q_type = classify_question(question)
    details = extract_event_details(question)
    
    # Initialize retrievers
    hybrid_retriever = HybridRetriever(tg_client)
    community_retriever = CommunityRetriever(tg_client)
    entity_retriever = EntityRelationshipRetriever(tg_client)
    
    # Build query based on question type
    if q_type == "temporal":
        target_year, target_season = resolve_temporal_reference(question)
        sport = details.get("sport", "")
        if target_year:
            search_query = f"{sport} at the {target_year} {target_season} Olympics gold medal"
        else:
            search_query = question
    elif q_type == "aggregation":
        sport = details.get("sport", "")
        year = details.get("year", 0)
        search_query = f"{sport} at the {year} Olympics events"
    elif q_type == "superlative":
        sport = details.get("sport", "")
        year = details.get("year", 0)
        search_query = f"{sport} at the {year} Olympics events athletes"
    elif q_type == "venue_date":
        venue = details.get("venue", "")
        year = details.get("year", 0)
        search_query = f"{venue} {year} Olympics"
    else:
        search_query = question
    
    # Multi-retriever GraphRAG: combine hybrid, community, and entity-based retrieval
    all_results = []
    
    # 1. Hybrid search (vector + graph expansion)
    hybrid_results = hybrid_retriever.search(search_query, top_k=8, num_hops=2)
    all_results.extend(hybrid_results)
    
    # 2. Community summaries (event-level context)
    community_results = community_retriever.search(search_query, top_k=5)
    all_results.extend(community_results)
    
    # 3. Entity-relationship retrieval
    entity_results = entity_retriever.search(search_query, top_k=5)
    all_results.extend(entity_results)
    
    # Deduplicate by doc_id
    seen = set()
    unique_results = []
    for r in all_results:
        if r.doc_id and r.doc_id not in seen:
            seen.add(r.doc_id)
            unique_results.append(r)
        elif not r.doc_id:
            unique_results.append(r)
    
    # Build context
    evidence = [r.content[:1200] for r in unique_results[:15]]
    context = "\n\n---\n\n".join(evidence) if evidence else "No relevant information found."
    
    # Type-specific instructions
    type_instructions = ""
    if q_type == "temporal":
        type_instructions = "\nThe previous Olympics year has been resolved. Find the gold medalist."
    elif q_type == "aggregation":
        threshold_match = re.search(r"more than (\d+)", question)
        if threshold_match:
            threshold = threshold_match.group(1)
            type_instructions = f"\nFor counting: count ALL events with more than {threshold} competitors. Give ONLY the number."
        else:
            type_instructions = "\nFor counting: count ALL matching events. Give ONLY the number."
    elif q_type == "superlative":
        type_instructions = "\nFor superlative: identify the SINGLE event with the highest/most. Give the event name."
    elif q_type == "venue_date":
        type_instructions = "\nFor venue+date questions: find the event at the specific venue on the given date."
    elif q_type == "lookup":
        type_instructions = "\nFor lookup: give the specific number or fact requested."
    
    prompt = f"""Answer this question using ONLY the provided evidence.
This evidence comes from hybrid graph+vector search, community summaries, and entity-relationship traversal.
Be precise and cite specific facts.{type_instructions}

Question: {question}

Evidence:
{context}

Answer:"""

    try:
        response = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
        total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
    except Exception as e:
        answer = f"Error: {e}"
    
    return {
        "pipeline": "GraphRAG",
        "question": question,
        "answer": answer,
        "results_count": len(unique_results),
        "total_tokens": total_tokens,
    }
