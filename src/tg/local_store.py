import json
import re
from typing import List, Dict, Optional


class LocalGraphStore:
    """In-memory graph store for local testing without TigerGraph."""

    def __init__(self):
        self.documents = {}
        self.events = {}
        self.persons = {}
        self.nations = {}
        self.medal_edges = []
        self.doc_event_edges = []

    def load_corpus(self, corpus_path: str):
        """Load corpus JSONL into local store."""
        print(f"Loading corpus from {corpus_path}...")
        count = 0
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc["doc_id"]
                self.documents[doc_id] = {
                    "doc_id": doc_id,
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "wikidata_qid": doc.get("wikidata_qid", ""),
                    "wikipedia_pageid": doc.get("wikipedia_pageid", 0),
                    "approx_tokens": doc.get("approx_tokens", 0),
                    "text": doc.get("text", ""),
                }
                self._extract_entities_from_doc(doc)
                count += 1
        print(f"Loaded {count} documents, {len(self.events)} events, {len(self.persons)} persons")

    def _extract_entities_from_doc(self, doc):
        """Extract events, persons, nations from a document."""
        title = doc.get("title", "")
        text = doc.get("text", "")
        doc_id = doc["doc_id"]

        year_match = re.search(r"(\d{4})\s+(Summer|Winter)\s+Olympics", title)
        if year_match:
            year = int(year_match.group(1))
            season = year_match.group(2)
            sport_match = re.match(r"^(.+?)\s+at\s+the", title)
            sport = sport_match.group(1).strip() if sport_match else ""
            event_id = f"{sport}_{year}_{season}"

            if event_id not in self.events:
                venue = ""
                venue_match = re.search(r"venue:\s*(.+)", text)
                if venue_match:
                    venue = venue_match.group(1).strip()

                self.events[event_id] = {
                    "event_id": event_id,
                    "name": title.split("–")[-1].strip() if "–" in title else title,
                    "sport": sport,
                    "year": year,
                    "season": season,
                    "venue": venue,
                }
            self.doc_event_edges.append({"doc_id": doc_id, "event_id": event_id})

        medal_map = {"gold": "Gold", "silver": "Silver", "bronze": "Bronze"}
        for medal_key, medal_type in medal_map.items():
            noc_pattern = rf"{medal_key}NOC:\s*(\w+)"
            noc_matches = re.findall(noc_pattern, text, re.IGNORECASE)

            name_pattern = rf"{medal_key}:\s*(.+?)(?:\n|{medal_key}NOC)"
            name_matches = re.findall(name_pattern, text, re.IGNORECASE)

            for i, match in enumerate(name_matches):
                noc = noc_matches[i] if i < len(noc_matches) else ""
                names = [n.strip() for n in match.split("\n") if n.strip()]

                if noc and noc not in self.nations:
                    self.nations[noc] = {"noc_code": noc, "name": noc}

                for name in names:
                    if len(name) > 2:
                        pid = re.sub(r"[^a-zA-Z0-9]", "_", name)[:50]
                        if pid not in self.persons:
                            self.persons[pid] = {
                                "person_id": pid,
                                "name": name,
                                "noc_code": noc,
                            }

                        if year_match:
                            medal_key_str = f"{pid}_{event_id}_{medal_type}"
                            if not any(
                                m["person_id"] == pid and m["event_id"] == event_id and m["medal_type"] == medal_type
                                for m in self.medal_edges
                            ):
                                self.medal_edges.append({
                                    "person_id": pid,
                                    "event_id": event_id,
                                    "medal_type": medal_type,
                                    "nation": noc,
                                })

    def search_documents(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search documents using improved keyword matching with phrase detection."""
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 2]

        scored = []
        for doc in self.documents.values():
            text_lower = doc["text"].lower()
            title_lower = doc["title"].lower()

            score = 0

            # Exact title match (highest weight)
            if query_lower in title_lower:
                score += 200

            # All query words in title
            words_in_title = sum(1 for w in query_words if w in title_lower)
            if words_in_title == len(query_words) and query_words:
                score += 100
            elif words_in_title > 0:
                score += words_in_title * 15

            # Exact phrase in text
            if query_lower in text_lower:
                score += 80

            # Word frequency in text
            for word in query_words:
                count = text_lower.count(word)
                score += min(count * 3, 30)

            # Partial word matches in title (e.g., "biathlon" matches "Biathlon at...")
            for word in query_words:
                if word in title_lower:
                    score += 20

            if score > 0:
                scored.append({"score": score, "doc": doc})

        scored.sort(key=lambda x: x["score"], reverse=True)
        max_score = scored[0]["score"] if scored else 1
        return [
            {**s["doc"], "score": s["score"] / max_score}
            for s in scored[:top_k]
        ]

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def search_events(self, query: str) -> List[Dict]:
        """Search events by name or sport."""
        query_lower = query.lower()
        results = []
        for ev in self.events.values():
            if (query_lower in ev["name"].lower() or
                query_lower in ev["sport"].lower() or
                query_lower in str(ev["year"])):
                results.append(ev)
        return results

    def get_events_by_sport_year(self, sport: str, year: int) -> List[Dict]:
        """Get all events for a specific sport and year."""
        sport_lower = sport.lower()
        return [
            ev for ev in self.events.values()
            if sport_lower in ev["sport"].lower() and ev["year"] == year
        ]

    def get_documents_by_event_prefix(self, title_prefix: str) -> List[Dict]:
        """Get all documents whose title starts with a prefix (e.g., 'Biathlon at the 2018')."""
        prefix_lower = title_prefix.lower()
        return [
            doc for doc in self.documents.values()
            if doc["title"].lower().startswith(prefix_lower)
        ]

    def search_persons(self, query: str) -> List[Dict]:
        """Search persons by name."""
        query_lower = query.lower()
        return [p for p in self.persons.values() if query_lower in p["name"].lower()]

    def get_medalists(self, event_id: str) -> List[Dict]:
        """Get medalists for an event."""
        return [m for m in self.medal_edges if m["event_id"] == event_id]

    def get_person_events(self, person_id: str) -> List[Dict]:
        """Get events a person participated in."""
        medals = [m for m in self.medal_edges if m["person_id"] == person_id]
        results = []
        for m in medals:
            ev = self.events.get(m["event_id"], {})
            results.append({**m, **ev})
        return results

    def get_docs_for_event(self, event_id: str) -> List[Dict]:
        """Get documents mentioning an event."""
        doc_ids = [e["doc_id"] for e in self.doc_event_edges if e["event_id"] == event_id]
        return [self.documents[did] for did in doc_ids if did in self.documents]

    def count_competitors(self, event_id: str) -> int:
        """Count unique competitors in an event."""
        persons = set()
        for m in self.medal_edges:
            if m["event_id"] == event_id:
                persons.add(m["person_id"])
        return len(persons)

    def traverse_from_person(self, person_id: str) -> List[Dict]:
        """Traverse from a person node."""
        medals = [m for m in self.medal_edges if m["person_id"] == person_id]
        results = []
        for m in medals:
            ev = self.events.get(m["event_id"], {})
            results.append({
                "person": self.persons.get(person_id, {}).get("name", person_id),
                "event": ev.get("name", m["event_id"]),
                "year": ev.get("year", ""),
                "sport": ev.get("sport", ""),
                "medal": m["medal_type"],
                "nation": m["nation"],
            })
        return results

    def traverse_from_event(self, event_id: str) -> List[Dict]:
        """Traverse from an event node."""
        medals = [m for m in self.medal_edges if m["event_id"] == event_id]
        results = []
        for m in medals:
            person = self.persons.get(m["person_id"], {})
            results.append({
                "person": person.get("name", m["person_id"]),
                "event": self.events.get(event_id, {}).get("name", event_id),
                "year": self.events.get(event_id, {}).get("year", ""),
                "medal": m["medal_type"],
                "nation": m["nation"],
            })
        return results
