import json
import re
from typing import Dict, List, Set


def parse_event_title(title: str) -> Dict:
    """Extract sport, year, season, event name from Wikipedia title."""
    result = {"sport": "", "year": 0, "season": "", "event_name": title}

    m = re.search(r"(\d{4})\s+(Summer|Winter)\s+Olympics", title)
    if m:
        result["year"] = int(m.group(1))
        result["season"] = m.group(2)

    sport_match = re.match(r"^(.+?)\s+at\s+the", title)
    if sport_match:
        result["sport"] = sport_match.group(1).strip()

    event_match = re.search(r"[–—]\s*(.+)$", title)
    if event_match:
        result["event_name"] = event_match.group(1).strip()

    return result


def extract_persons_from_text(text: str) -> List[Dict]:
    """Extract person names and medals from infobox text."""
    persons = []
    medal_map = {"gold": "Gold", "silver": "Silver", "bronze": "Bronze"}

    for medal_key, medal_type in medal_map.items():
        pattern = rf"{medal_key}:\s*(.+?)(?:\n|{medal_key}NOC)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        noc_pattern = rf"{medal_key}NOC:\s*(\w+)"

        noc_matches = re.findall(noc_pattern, text, re.IGNORECASE)

        for i, match in enumerate(matches):
            names = [n.strip() for n in re.split(r"(?<=[a-z])(?=[A-Z])", match) if n.strip()]
            noc = noc_matches[i] if i < len(noc_matches) else ""
            for name in names:
                if len(name) > 2:
                    persons.append({"name": name, "medal": medal_type, "noc": noc})

    return persons


def extract_nations_from_text(text: str) -> List[str]:
    """Extract nation/NOC codes from text."""
    noc_match = re.findall(r"\b([A-Z]{3})\b", text)
    return list(set(noc_match))


def generate_load_queries(corpus_path: str) -> Dict[str, List[str]]:
    """Generate GSQL INSERT statements from corpus JSONL."""
    vertices = []
    edges = []
    doc_ids = set()
    event_ids = set()
    person_ids = set()
    nation_codes = set()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            title = doc.get("title", "")
            text = doc.get("text", "")

            doc_ids.add(doc_id)

            safe_text = text.replace("'", "\\'").replace('"', '\\"')[:5000]
            safe_title = title.replace("'", "\\'").replace('"', '\\"')

            vertices.append(
                f"INSERT INTO Document VALUES (\"{doc_id}\", \"{safe_title}\", "
                f"\"{doc.get('url', '')}\", \"{doc.get('wikidata_qid', '')}\", "
                f"{doc.get('wikipedia_pageid', 0)}, {doc.get('approx_tokens', 0)}, "
                f"\"{safe_text}\")"
            )

            parsed = parse_event_title(title)
            if parsed["year"] > 0:
                event_id = f"{parsed['sport']}_{parsed['year']}_{parsed['season']}"
                event_ids.add(event_id)

                if event_id not in event_ids:
                    safe_event_name = parsed["event_name"].replace("'", "\\'")
                    safe_sport = parsed["sport"].replace("'", "\\'")
                    safe_venue = ""
                    venue_match = re.search(r"venue:\s*(.+)", text)
                    if venue_match:
                        safe_venue = venue_match.group(1).strip().replace("'", "\\'")

                    vertices.append(
                        f"INSERT INTO Event VALUES (\"{event_id}\", \"{safe_event_name}\", "
                        f"\"{safe_sport}\", {parsed['year']}, \"{parsed['season']}\", "
                        f"\"{safe_venue}\", \"\")"
                    )

                edges.append(
                    f"INSERT INTO doc_mentions_event VALUES (\"{doc_id}\", \"{event_id}\")"
                )

            persons = extract_persons_from_text(text)
            for p in persons:
                pid = p["name"].replace(" ", "_").replace("'", "")
                if pid not in person_ids:
                    person_ids.add(pid)
                    safe_name = p["name"].replace("'", "\\'")
                    vertices.append(
                        f"INSERT INTO Person VALUES (\"{pid}\", \"{safe_name}\", "
                        f"\"\", \"{p['noc']}\")"
                    )

                if p["noc"] and p["noc"] not in nation_codes:
                    nation_codes.add(p["noc"])
                    vertices.append(
                        f"INSERT INTO Nation VALUES (\"{p['noc']}\", \"{p['noc']}\")"
                    )

                if parsed["year"] > 0:
                    event_id = f"{parsed['sport']}_{parsed['year']}_{parsed['season']}"
                    edges.append(
                        f"INSERT INTO won_medal VALUES (\"{pid}\", \"{event_id}\", "
                        f"\"{p['medal']}\", \"{p['noc']}\")"
                    )

    return {"vertices": vertices, "edges": edges}


def save_gsql_inserts(queries: Dict[str, List[str]], output_dir: str):
    """Save GSQL INSERT statements to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "insert_vertices.gsql"), "w") as f:
        f.write("\n".join(queries["vertices"]))

    with open(os.path.join(output_dir, "insert_edges.gsql"), "w") as f:
        f.write("\n".join(queries["edges"]))

    print(f"Generated {len(queries['vertices'])} vertex inserts")
    print(f"Generated {len(queries['edges'])} edge inserts")
