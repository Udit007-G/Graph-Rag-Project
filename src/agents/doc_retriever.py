from typing import List, Dict


def get_document(doc_id: str, client=None) -> Dict:
    """Retrieve a full document by doc_id."""
    is_local = hasattr(client, "get_document") if client else False

    if is_local:
        return client.get_document(doc_id) or {}

    query = f"""SELECT d.doc_id, d.title, d.url, d.text, d.approx_tokens
                FROM Document d WHERE d.doc_id == "{doc_id}" """
    try:
        result = client.runQuery(query)
        if result and "results" in result:
            data_key = list(result["results"].keys())[0] if result["results"] else None
            if data_key:
                for row in result["results"][data_key]:
                    return {
                        "doc_id": row.get("d.doc_id", ""),
                        "title": row.get("d.title", ""),
                        "url": row.get("d.url", ""),
                        "text": row.get("d.text", ""),
                        "tokens": row.get("d.approx_tokens", 0)
                    }
    except Exception as e:
        print(f"Document retrieval error: {e}")
    return {}


def get_documents(doc_ids: List[str], client=None) -> List[Dict]:
    """Retrieve multiple documents by their IDs."""
    return [doc for doc_id in doc_ids if (doc := get_document(doc_id, client))]


def get_medalists_for_event(event_id: str, client=None) -> List[Dict]:
    """Get all medalists for a specific event."""
    is_local = hasattr(client, "get_medalists") if client else False

    if is_local:
        return client.get_medalists(event_id)

    query = f"""SELECT p.name AS person_name, w.medal_type AS medal,
                w.nation AS nation, e.name AS event_name, e.year AS event_year
                FROM (SELECT p2, medal_type, nation FROM won_medal
                      WHERE TO_ID == "{event_id}") w, Person p2, Event e
                WHERE w.p2 == p2.person_id AND e.event_id == "{event_id}" """
    medalists = []
    try:
        result = client.runQuery(query)
        if result and "results" in result:
            data_key = list(result["results"].keys())[0] if result["results"] else None
            if data_key:
                medalists = [{
                    "person": row.get("p2.name", row.get("person_name", "")),
                    "medal": row.get("w.medal_type", row.get("medal", "")),
                    "nation": row.get("w.nation", row.get("nation", "")),
                    "event": row.get("e.name", row.get("event_name", "")),
                    "year": row.get("e.year", row.get("event_year", ""))
                } for row in result["results"][data_key]]
    except Exception as e:
        print(f"Medalist query error: {e}")
    return medalists


def count_competitors(event_id: str, client=None) -> int:
    """Count competitors in an event."""
    is_local = hasattr(client, "count_competitors") if client else False

    if is_local:
        return client.count_competitors(event_id)

    query = f"""SELECT COUNT(p.person_id) AS cnt
                FROM (SELECT p2 FROM participated_in WHERE TO_ID == "{event_id}") pi, Person p2
                WHERE pi.p2 == p2.person_id"""
    try:
        result = client.runQuery(query)
        if result and "results" in result:
            data_key = list(result["results"].keys())[0] if result["results"] else None
            if data_key:
                for row in result["results"][data_key]:
                    return row.get("cnt", 0)
    except Exception:
        pass
    return 0
