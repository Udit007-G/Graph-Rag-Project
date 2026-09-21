"""Entity extraction from questions and linking to graph nodes."""
import json
import re
from typing import List, Dict
from config.settings import llm_completion


def extract_entities(question: str) -> List[Dict]:
    """Extract entities (persons, events, nations, years) from the question."""
    prompt = f"""Extract all named entities from this question about Olympic sports.
Return a JSON array of objects with "text" and "type" fields.
Types can be: person, event, sport, nation, year, venue, other.

Question: {question}

Return ONLY valid JSON array, no other text."""

    try:
        response = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "entities" in result:
            return result["entities"]
        return []
    except Exception as e:
        print(f"Entity extraction error: {e}")
        return _extract_entities_fallback(question)


def _extract_entities_fallback(question: str) -> List[Dict]:
    """Simple regex-based entity extraction fallback."""
    entities = []

    year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    for y in year_matches:
        entities.append({"text": y, "type": "year"})

    if "gold medal" in question.lower():
        entities.append({"text": "gold medal", "type": "medal"})

    nation_pattern = r"\b([A-Z]{3})\b"
    for m in re.findall(nation_pattern, question):
        if m not in ("THE", "AND", "FOR", "WHO", "HOW", "DID", "WAS", "HAS"):
            entities.append({"text": m, "type": "nation"})

    return entities


def link_entities_to_graph(entities: List[Dict], client) -> List[Dict]:
    """Link extracted entities to graph nodes."""
    linked = []
    is_local = hasattr(client, "search_events")

    for entity in entities:
        text = entity.get("text", "")
        etype = entity.get("type", "other")

        if etype == "person" or (etype == "other" and len(text.split()) >= 2):
            if is_local:
                results = client.search_persons(text)
                for p in results[:3]:
                    linked.append({
                        "text": text, "type": "person",
                        "graph_id": p["person_id"], "graph_name": p["name"],
                        "noc": p.get("noc_code", ""),
                    })
            else:
                query = f"""SELECT p.name, p.person_id, p.noc_code
                            FROM Person p WHERE p.name CONTAINS '{text.replace("'", "''")}' LIMIT 3"""
                try:
                    result = client.runQuery(query)
                    if result and "results" in result:
                        for row in result["results"].get("p", []):
                            linked.append({
                                "text": text, "type": "person",
                                "graph_id": row.get("p.person_id", ""),
                                "graph_name": row.get("p.name", ""),
                                "noc": row.get("p.noc_code", ""),
                            })
                except Exception:
                    pass

        elif etype in ("year", "sport", "event"):
            if is_local:
                results = client.search_events(text)
                for ev in results[:5]:
                    linked.append({
                        "text": text, "type": "event",
                        "graph_id": ev["event_id"], "graph_name": ev["name"],
                    })
            else:
                query = f"""SELECT e.event_id, e.name, e.sport, e.year
                            FROM Event e
                            WHERE e.year = {text} OR e.sport CONTAINS '{text.replace("'", "''")}'
                            LIMIT 5"""
                try:
                    result = client.runQuery(query)
                    if result and "results" in result:
                        for row in result["results"].get("e", []):
                            linked.append({
                                "text": text, "type": "event",
                                "graph_id": row.get("e.event_id", ""),
                                "graph_name": row.get("e.name", ""),
                            })
                except Exception:
                    pass

    return linked
