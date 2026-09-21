"""Shared utilities for question classification, temporal resolution, and detail extraction."""
import re
from typing import Dict, Optional, Tuple


OLYMPICS_YEAR_MAP = {
    2022: ("Winter", 2018), 2020: ("Summer", 2016), 2018: ("Winter", 2014),
    2016: ("Summer", 2012), 2014: ("Winter", 2010), 2012: ("Summer", 2008),
    2010: ("Winter", 2006), 2008: ("Summer", 2004), 2006: ("Winter", 2002),
    2004: ("Summer", 2000), 2002: ("Winter", 1998), 2000: ("Summer", 1996),
    1998: ("Winter", 1994), 1996: ("Summer", 1992), 1994: ("Winter", 1992),
    1992: ("Summer", 1988), 1988: ("Summer", 1984), 1984: ("Summer", 1980),
}

SPORT_KEYWORDS = [
    "biathlon", "athletics", "swimming", "cycling", "shooting", "fencing",
    "judo", "wrestling", "boxing", "rowing", "sailing", "canoeing",
    "weightlifting", "gymnastics", "skating", "skiing", "tennis", "volleyball",
    "diving", "soccer", "football", "archery", "taekwondo", "triathlon",
    "equestrian", "handball", "basketball", "golf", "kayak", "bobsled",
    "luge", "skeleton", "curling", "ice hockey", "figure skating",
    "short-track", "speed skating", "nordic", "alpine", "freestyle",
    "snowboard", "cross-country",
]


def classify_question(question: str) -> str:
    """Classify question type for strategy selection."""
    q_lower = question.lower()

    if "held immediately before" in q_lower:
        return "temporal"
    elif "how many" in q_lower and ("had more than" in q_lower or "highest" in q_lower):
        return "aggregation"
    elif "how many nations" in q_lower or "how many competitors" in q_lower:
        return "lookup"
    elif "highest number of competitors" in q_lower or "most competitors" in q_lower:
        return "superlative"
    elif "held at" in q_lower and ("on" in q_lower or "august" in q_lower or "february" in q_lower):
        return "venue_date"
    elif "who won" in q_lower or "gold medal" in q_lower:
        return "who_won"
    return "general"


def resolve_temporal_reference(question: str) -> Tuple[Optional[int], Optional[int]]:
    """Resolve temporal reference in question to (target_year, target_season).

    Handles both 'held immediately before YYYY' and direct year references like
    'at the 2016 Summer Olympics' or 'at the 2010 Winter Olympics'.
    """
    match = re.search(r"held immediately before (\d{4})", question)
    if match:
        ref_year = int(match.group(1))
        season_match = re.search(
            r"(Summer|Winter)\s+Olympics\s+held immediately before", question, re.IGNORECASE
        )
        if season_match:
            specified_season = season_match.group(1)
            prev_year = ref_year - 4
            return prev_year, specified_season
        else:
            prev_year = ref_year - 2
            if prev_year % 4 != 0:
                prev_year = ref_year - 4 if ref_year % 2 == 0 else ref_year - 3
            season = "Summer" if prev_year % 4 == 0 and prev_year % 2 == 0 else "Winter"
            return prev_year, season

    direct_match = re.search(r"(\d{4})\s+(Summer|Winter)\s+Olympics", question, re.IGNORECASE)
    if direct_match:
        return int(direct_match.group(1)), direct_match.group(2)

    bare_year = re.search(r"on\s+\d+\s+\w+\s+(\d{4})", question)
    if bare_year:
        year = int(bare_year.group(1))
        season = "Summer" if year % 4 == 0 else "Winter"
        return year, season

    return None, None


def extract_event_details(question: str) -> Dict:
    """Extract sport, event name, year, season, venue, date from question."""
    details = {}

    year_match = re.search(r"(\d{4})\s+(Summer|Winter)\s+Olympics", question, re.IGNORECASE)
    if year_match:
        details["year"] = int(year_match.group(1))
        details["season"] = year_match.group(2)

    held_match = re.search(r"held immediately before (\d{4})", question)
    if held_match:
        target_year, target_season = resolve_temporal_reference(question)
        if target_year:
            details["year"] = target_year
            details["season"] = target_season

    q_lower = question.lower()
    for kw in SPORT_KEYWORDS:
        if kw in q_lower:
            details["sport"] = kw.title()
            break

    event_match = re.search(
        r"(?:men'?s?|women'?s?)\s+(.+?)\s+(?:event|at)", question, re.IGNORECASE
    )
    if event_match:
        details["event_text"] = event_match.group(0).strip()

    venue_match = re.search(r"held at\s+(.+?)\s+on\s+", question, re.IGNORECASE)
    if venue_match:
        details["venue"] = venue_match.group(1).strip()

    date_match = re.search(r"on\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
    if date_match:
        details["date_text"] = date_match.group(1).strip()

    return details


def extract_gold_from_text(text: str) -> Optional[str]:
    """Extract gold medalist name from document text."""
    gold_match = re.search(r"gold:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if gold_match:
        return gold_match.group(1).strip()
    return None


def extract_competitor_count(text: str) -> Optional[int]:
    """Extract competitor count from document text."""
    comp_match = re.search(r"competitors:\s*(\d+)", text, re.IGNORECASE)
    if comp_match:
        return int(comp_match.group(1))
    return None


def extract_nation_count(text: str) -> Optional[int]:
    """Extract nation count from document text."""
    nations_match = re.search(r"nations:\s*(\d+)", text, re.IGNORECASE)
    if nations_match:
        return int(nations_match.group(1))
    return None
