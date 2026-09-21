from typing import List, Dict


def traverse_from_entity(entity_id: str, entity_type: str, client=None) -> List[Dict]:
    """Traverse the graph from an entity, following edges."""
    is_local = hasattr(client, "search_events") if client else False

    if is_local:
        if entity_type == "person":
            return client.traverse_from_person(entity_id)
        elif entity_type == "event":
            return client.traverse_from_event(entity_id)
        return []

    if entity_type == "person":
        query = f"""SELECT p.name AS person_name, e.name AS event_name,
                    e.year AS event_year, e.sport AS event_sport,
                    w.medal_type AS medal, w.nation AS nation
                    FROM (SELECT e2, medal_type, nation FROM won_medal
                          WHERE FROM_ID == "{entity_id}") w, Event e2, Person p
                    WHERE w.e2 == e2.event_id AND p.person_id == "{entity_id}" """
    elif entity_type == "event":
        query = f"""SELECT p.name AS person_name, e.name AS event_name,
                    e.year AS event_year, w.medal_type AS medal, w.nation AS nation
                    FROM (SELECT p2, medal_type, nation FROM won_medal
                          WHERE TO_ID == "{entity_id}") w, Person p2, Event e
                    WHERE w.p2 == p2.person_id AND e.event_id == "{entity_id}" """
    else:
        return []

    try:
        result = client.runQuery(query)
        if result and "results" in result:
            data_key = list(result["results"].keys())[0] if result["results"] else None
            if data_key:
                return [{
                    "person": row.get("p.name", row.get("person_name", "")),
                    "event": row.get("e.name", row.get("event_name", "")),
                    "year": row.get("e.year", row.get("event_year", "")),
                    "sport": row.get("e.sport", row.get("event_sport", "")),
                    "medal": row.get("w.medal_type", row.get("medal", "")),
                    "nation": row.get("w.nation", row.get("nation", ""))
                } for row in result["results"][data_key]]
    except Exception as e:
        print(f"Graph traversal error: {e}")
    return []


def multi_hop_traverse(start_id: str, start_type: str, hops: int = 3, client=None) -> List[Dict]:
    """Perform multi-hop traversal from a starting entity."""
    all_results = []
    current_ids = [(start_id, start_type)]

    for hop in range(hops):
        next_ids = []
        for eid, etype in current_ids:
            results = traverse_from_entity(eid, etype, client=client)
            all_results.extend(results)
            for r in results:
                if r.get("person"):
                    name = r["person"]
                    if name not in [x[0] for x in next_ids]:
                        next_ids.append((name, "person"))
        current_ids = next_ids
        if not current_ids:
            break

    return all_results


def find_venue_docs(venue_name: str, client=None) -> List[str]:
    """Find documents related to a venue."""
    is_local = hasattr(client, "search_documents") if client else False

    if is_local:
        results = client.search_documents(venue_name, top_k=10)
        return [r["doc_id"] for r in results]

    query = f"""SELECT d.doc_id FROM Document d
                WHERE d.text CONTAINS '{venue_name.replace("'", "''")}' LIMIT 10"""
    doc_ids = []
    try:
        result = client.runQuery(query)
        if result and "results" in result:
            data_key = list(result["results"].keys())[0] if result["results"] else None
            if data_key:
                doc_ids = [row.get("d.doc_id", "") for row in result["results"][data_key]]
    except Exception:
        pass
    return doc_ids
