from typing import List, Dict
from config.settings import get_openai_client, get_model_name


def vector_search(query: str, top_k: int = 10, client=None) -> List[Dict]:
    """Search documents using keyword similarity."""
    is_local = hasattr(client, "search_documents") if client else False

    if is_local:
        return client.search_documents(query, top_k=top_k)

    query_gsql = f"""SELECT d.doc_id, d.title, d.text, d.approx_tokens
                     FROM Document d
                     WHERE d.text LIKE '%{query.replace("'", "''")}%'
                     LIMIT {top_k}"""
    results = []
    try:
        result = client.runQuery(query_gsql)
        if result and "results" in result:
            data_key = list(result["results"].keys())[0] if result["results"] else None
            if data_key:
                results = [{
                    "doc_id": row.get("d.doc_id", ""),
                    "title": row.get("d.title", ""),
                    "text": row.get("d.text", ""),
                    "score": 1.0
                } for row in result["results"][data_key]]
    except Exception as e:
        print(f"Vector search error: {e}")
    return results


def hybrid_search(query: str, entities: List[Dict] = None, top_k: int = 10, client=None) -> List[Dict]:
    """Combine keyword search with entity-aware graph lookup."""
    keyword_results = vector_search(query, top_k=top_k, client=client)
    entity_results = []

    if entities:
        is_local = hasattr(client, "search_events") if client else False
        for ent in entities:
            eid = ent.get("graph_id", "")
            if eid:
                if is_local:
                    ev_docs = client.get_docs_for_event(eid)
                    for d in ev_docs[:5]:
                        entity_results.append({
                            "doc_id": d["doc_id"], "title": d["title"],
                            "text": d["text"][:2000], "score": 0.8
                        })
                else:
                    query_gsql = f"""SELECT d.doc_id, d.title, d.text
                                    FROM (SELECT d2 FROM doc_mentions_event
                                          WHERE FROM_ID == "{eid}") de, Document d2
                                    WHERE de.d2 == d2.doc_id LIMIT 5"""
                    try:
                        result = client.runQuery(query_gsql)
                        if result and "results" in result:
                            data_key = list(result["results"].keys())[0] if result["results"] else None
                            if data_key:
                                for row in result["results"][data_key]:
                                    entity_results.append({
                                        "doc_id": row.get("d2.doc_id", ""),
                                        "title": row.get("d2.title", ""),
                                        "text": row.get("d2.text", ""),
                                        "score": 0.8
                                    })
                    except Exception:
                        pass

    seen_ids = set()
    combined = []
    for r in keyword_results + entity_results:
        if r["doc_id"] not in seen_ids:
            seen_ids.add(r["doc_id"])
            combined.append(r)

    return combined[:top_k]
