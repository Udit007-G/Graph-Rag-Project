"""Agentic orchestrator with plan-execute-evaluate-replan loop."""
import json
import re
from typing import Dict, List, Optional
from config.settings import get_openai_client, get_model_name, MAX_STEPS, llm_completion
from src.agents.state import InvestigationState, Action, Evidence
from src.agents.entity_linker import extract_entities, link_entities_to_graph
from src.agents.graph_traverser import traverse_from_entity
from src.agents.similarity_search import hybrid_search
from src.agents.doc_retriever import get_document
from src.utils.helpers import (
    classify_question, resolve_temporal_reference, extract_event_details,
    extract_gold_from_text, extract_competitor_count, extract_nation_count,
)


def plan_investigation(question: str, state: InvestigationState) -> List[str]:
    """LLM generates an initial investigation plan based on the question."""
    model = get_model_name()

    prompt = f"""You are an investigation planner for Olympic data questions.
Given the question below, produce a JSON list of investigation steps (actions to take).
Each step should be one of: entity_link, graph_traverse, similarity_search, document_retrieve, aggregate, synthesize_answer.

Question: {question}

Available information sources:
- Entity linking: extract persons, events, sports, nations, years from the question
- Graph traversal: follow edges from known entities to find related data
- Similarity search: find documents matching keywords
- Document retrieve: get full text of a specific document by ID
- Aggregate: count/filter/combine existing evidence
- Synthesize answer: produce final answer (only when enough evidence collected)

Return ONLY a JSON array of step names. Example: ["entity_link", "similarity_search", "graph_traverse", "aggregate", "synthesize_answer"]
"""

    try:
        response = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "steps" in result:
            return result["steps"]
        return ["entity_link", "similarity_search", "graph_traverse", "synthesize_answer"]
    except Exception:
        return ["entity_link", "similarity_search", "graph_traverse", "synthesize_answer"]


def evaluate_evidence(question: str, state: InvestigationState) -> Dict:
    """LLM evaluates whether collected evidence is sufficient to answer."""
    model = get_model_name()

    evidence_texts = []
    for ev in state.evidence[-8:]:
        evidence_texts.append(f"[{ev.source}]: {ev.content[:400]}")
    combined = "\n".join(evidence_texts) if evidence_texts else "No evidence collected yet."

    prompt = f"""Evaluate if the following evidence is sufficient to answer the question.
Return a JSON object:
- "sufficient": boolean
- "confidence": float (0-1)
- "gaps": list of strings describing what is missing
- "next_action": one of [entity_link, graph_traverse, similarity_search, document_retrieve, aggregate, synthesize_answer]

Question: {question}

Evidence so far:
{combined}

Evaluation:"""

    try:
        response = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=300,
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return result
    except Exception:
        return {
            "sufficient": len(state.evidence) >= 3,
            "confidence": 0.5,
            "gaps": [],
            "next_action": "synthesize_answer",
        }


def synthesize_answer(question: str, evidence_texts: list, state: InvestigationState) -> str:
    """Generate final answer from collected evidence."""
    model = get_model_name()

    priority_sources = {"agg", "agg_result", "temporal", "lookup_result", "superlative_result"}
    priority_evidence = [ev for ev in state.evidence
                         if any(ev.source.startswith(s) for s in priority_sources)]
    other_evidence = [ev for ev in state.evidence
                      if not any(ev.source.startswith(s) for s in priority_sources)]

    priority_parts = []
    for ev in priority_evidence:
        priority_parts.append(f"[PRIORITY EVIDENCE] {ev.content[:1500]}")

    other_parts = []
    total_len = sum(len(p) for p in priority_parts)
    for ev in other_evidence:
        chunk = ev.content[:1000]
        if total_len + len(chunk) < 6000:
            other_parts.append(f"[{ev.source}] {chunk}")
            total_len += len(chunk)

    combined = "\n\n".join(priority_parts + other_parts) if priority_parts else "\n\n".join(other_parts[:8])

    q_type = classify_question(question)

    type_instructions = ""
    if q_type == "aggregation":
        type_instructions = "\nFor counting questions: The AGGREGATE RESULT section contains the pre-computed count. Use that number directly."
    elif q_type == "lookup":
        type_instructions = "\nFor lookup questions: give the specific number or fact requested."
    elif q_type == "superlative":
        type_instructions = "\nFor superlative questions: The AGGREGATE RESULT section contains the answer. Use it directly."
    elif q_type == "temporal":
        type_instructions = "\nFor temporal questions: carefully resolve 'held immediately before YYYY' to the correct Olympics, then find the gold medalist."
    elif q_type == "venue_date":
        type_instructions = "\nFor venue+date questions: find the event at the specific venue on the given date, then identify the gold medalist."

    prompt = f"""Answer this question using ONLY the provided evidence.
Be precise and cite specific facts. If the evidence is insufficient, say so.{type_instructions}

IMPORTANT: PRIORITY EVIDENCE sections contain the most relevant facts. Use them directly.

Question: {question}

Evidence:
{combined}

Answer:"""

    try:
        response = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
        state.add_token_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"


# ─── Action Executors ───────────────────────────────────────────────────────

def _execute_entity_link(question: str, state: InvestigationState, client) -> None:
    """Extract entities from question and link to graph."""
    entities = extract_entities(question)
    state.record_action(Action.ENTITY_LINK, {"entities": entities})

    linked = link_entities_to_graph(entities, client)
    state.visited_entities.extend([e.get("text", "") for e in linked])
    state.record_action(Action.ENTITY_LINK, {"linked": linked})

    for ent in linked:
        eid = ent.get("graph_id", "")
        etype = ent.get("type", "other")
        if eid:
            state.add_evidence(Evidence(
                source=f"entity({etype}:{eid})",
                content=f"Linked entity: {ent.get('text', '')} -> {ent.get('graph_name', eid)} (type={etype})",
                step_obtained=state.step_count,
            ))


def _execute_graph_traverse(state: InvestigationState, client) -> None:
    """Traverse graph from all linked entities."""
    is_local = hasattr(client, "traverse_from_person")

    for ent_text in state.visited_entities[:5]:
        for ent in state.actions_taken:
            if ent.get("action") == "entity_link" and "linked" in ent.get("details", {}):
                for linked_ent in ent["details"]["linked"]:
                    if linked_ent.get("text") == ent_text:
                        eid = linked_ent.get("graph_id", "")
                        etype = linked_ent.get("type", "person")
                        if eid:
                            results = traverse_from_entity(eid, etype, client=client)
                            if results:
                                lines = []
                                for r in results[:15]:
                                    lines.append(
                                        f"{r.get('person', '?')} - {r.get('medal', '?')} - "
                                        f"{r.get('event', '?')} ({r.get('year', '?')}) - "
                                        f"{r.get('nation', '?')}"
                                    )
                                summary = "\n".join(lines)
                                state.add_evidence(Evidence(
                                    source=f"graph({etype}:{eid})",
                                    content=summary,
                                    step_obtained=state.step_count,
                                ))
    state.record_action(Action.GRAPH_TRAVERSE, {"evidence_count": len(state.evidence)})


def _execute_similarity_search(question: str, state: InvestigationState, client) -> None:
    """Search for relevant documents."""
    linked = []
    for act in state.actions_taken:
        if act.get("action") == "entity_link" and "linked" in act.get("details", {}):
            linked.extend(act["details"]["linked"])

    results = hybrid_search(question, entities=linked, top_k=8, client=client)
    for r in results[:8]:
        text = r.get("text", "")[:1200]
        if text:
            state.add_evidence(Evidence(
                source=f"search({r.get('doc_id', '')})",
                content=f"Title: {r.get('title', '')}\n{text}",
                doc_ids=[r.get("doc_id", "")],
                step_obtained=state.step_count,
            ))
    state.record_action(Action.SIMILARITY_SEARCH, {"results": len(results)})


def _execute_aggregate(question: str, state: InvestigationState, client) -> None:
    """Count/filter/combine evidence based on question type."""
    q_type = classify_question(question)
    details = extract_event_details(question)
    sport = details.get("sport", "")
    year = details.get("year", 0)
    is_local = hasattr(client, "get_events_by_sport_year")

    if q_type == "aggregation":
        threshold_match = re.search(r"more than (\d+)", question)
        threshold = int(threshold_match.group(1)) if threshold_match else 0

        if is_local and sport and year:
            events = client.get_events_by_sport_year(sport, year)
            count = 0
            matching = []
            for ev in events:
                docs = client.get_docs_for_event(ev["event_id"])
                for doc in docs:
                    comp = extract_competitor_count(doc.get("text", "")[:500])
                    if comp and comp > threshold:
                        count += 1
                        matching.append(f"{ev.get('name', ev['event_id'])}: {comp}")
                        state.add_evidence(Evidence(
                            source=f"agg({ev['event_id']})",
                            content=f"Event: {ev.get('name', ev['event_id'])}\nCompetitors: {comp} (>{threshold})",
                            doc_ids=[doc["doc_id"]],
                            step_obtained=state.step_count,
                        ))
            state.add_evidence(Evidence(
                source="agg_result",
                content=f"Count: {count} {sport} events at {year} with >{threshold} competitors:\n" + "\n".join(matching),
                step_obtained=state.step_count,
            ))
        else:
            count = 0
            matching = []
            existing = list(state.evidence)
            for ev in existing:
                comp = extract_competitor_count(ev.content)
                if comp and comp > threshold:
                    count += 1
                    matching.append(ev.content[:100])
            state.add_evidence(Evidence(
                source="agg_result",
                content=f"Count from evidence: {count} events with >{threshold} competitors:\n" + "\n".join(matching),
                step_obtained=state.step_count,
            ))

    elif q_type == "superlative":
        if is_local and sport and year:
            events = client.get_events_by_sport_year(sport, year)
            best_name, best_count = "", 0
            all_events = []
            for ev in events:
                docs = client.get_docs_for_event(ev["event_id"])
                for doc in docs:
                    comp = extract_competitor_count(doc.get("text", "")[:500])
                    if comp:
                        all_events.append((ev.get("name", ev["event_id"]), comp))
                        if comp > best_count:
                            best_count = comp
                            best_name = ev.get("name", ev["event_id"])
            if best_name:
                state.add_evidence(Evidence(
                    source="superlative_result",
                    content=f"Highest: {best_name} ({best_count} competitors)\nAll: " +
                            ", ".join(f"{n}:{c}" for n, c in sorted(all_events, key=lambda x: -x[1])),
                    step_obtained=state.step_count,
                ))
        else:
            best_name, best_count = "", 0
            existing = list(state.evidence)
            for ev in existing:
                comp = extract_competitor_count(ev.content)
                if comp and comp > best_count:
                    best_count = comp
                    best_name = ev.content[:100]
            if best_name:
                state.add_evidence(Evidence(
                    source="superlative_result",
                    content=f"Highest from evidence: {best_name} ({best_count} competitors)",
                    step_obtained=state.step_count,
                ))

    elif q_type == "lookup":
        existing = list(state.evidence)
        for ev in existing:
            nation_count = extract_nation_count(ev.content)
            if nation_count:
                state.add_evidence(Evidence(
                    source="lookup_result",
                    content=f"Nations: {nation_count}",
                    step_obtained=state.step_count,
                ))

    state.record_action(Action.AGGREGATE, {"type": q_type})


def _execute_temporal(question: str, state: InvestigationState, client) -> None:
    """Handle temporal questions by searching for the resolved year."""
    target_year, target_season = resolve_temporal_reference(question)
    if not target_year:
        return

    details = extract_event_details(question)
    sport = details.get("sport", "")
    event_text = details.get("event_text", "")
    venue = details.get("venue", "")
    is_local = hasattr(client, "search_events")

    if is_local:
        events = client.search_events(str(target_year))
        if sport:
            events = [e for e in events if sport.lower() in e.get("sport", "").lower()]

        found = False
        for ev in events[:20]:
            docs = client.get_docs_for_event(ev["event_id"])
            for doc in docs:
                text = doc.get("text", "")
                gold = extract_gold_from_text(text)
                title = doc.get("title", "").lower()
                doc_text = (text[:2000] + " " + title).lower()
                if gold and venue and venue.lower() in doc_text:
                    state.add_evidence(Evidence(
                        source=f"temporal({ev['event_id']})",
                        content=f"Event: {doc.get('title', ev.get('name', ''))}\nGold: {gold}\nYear: {target_year}",
                        doc_ids=[doc["doc_id"]],
                        step_obtained=state.step_count,
                    ))
                    found = True
                    break
                elif gold and event_text:
                    event_words = [w for w in event_text.lower().split()
                                   if len(w) > 3 and w not in ("men's", "women's", "men", "women", "at")]
                    if event_words and all(w in title for w in event_words):
                        state.add_evidence(Evidence(
                            source=f"temporal({ev['event_id']})",
                            content=f"Event: {doc.get('title', ev.get('name', ''))}\nGold: {gold}\nYear: {target_year}",
                            doc_ids=[doc["doc_id"]],
                            step_obtained=state.step_count,
                        ))
                        found = True
                        break
            if found:
                break

        if not found:
            for ev in events[:20]:
                docs = client.get_docs_for_event(ev["event_id"])
                for doc in docs:
                    gold = extract_gold_from_text(doc.get("text", ""))
                    if gold:
                        state.add_evidence(Evidence(
                            source=f"temporal({ev['event_id']})",
                            content=f"Event: {doc.get('title', ev.get('name', ''))}\nGold: {gold}\nYear: {target_year}",
                            doc_ids=[doc["doc_id"]],
                            step_obtained=state.step_count,
                        ))
    else:
        search_query = f"{venue} {target_year} {target_season} Olympics gold medal" if venue else \
                       f"{event_text} at the {target_year} {target_season} Olympics".strip()
        results = hybrid_search(search_query, top_k=15, client=client)

        found = False
        for r in results[:15]:
            gold = extract_gold_from_text(r.get("text", ""))
            title = r.get("title", "").lower()
            doc_text = (r.get("text", "")[:2000] + " " + title).lower()
            if gold and venue and venue.lower() in doc_text:
                state.add_evidence(Evidence(
                    source=f"temporal_doc({r.get('doc_id', '')})",
                    content=f"Title: {r.get('title', '')}\nGold: {gold}",
                    doc_ids=[r.get("doc_id", "")],
                    step_obtained=state.step_count,
                ))
                found = True
                break

        if not found:
            for r in results[:10]:
                gold = extract_gold_from_text(r.get("text", ""))
                if gold:
                    state.add_evidence(Evidence(
                        source=f"temporal_doc({r.get('doc_id', '')})",
                        content=f"Title: {r.get('title', '')}\nGold: {gold}",
                        doc_ids=[r.get("doc_id", "")],
                        step_obtained=state.step_count,
                    ))


# ─── Action Dispatcher ──────────────────────────────────────────────────────

def execute_action(action: str, question: str, state: InvestigationState, client) -> None:
    """Execute a single investigation action."""
    if action == "entity_link":
        _execute_entity_link(question, state, client)
    elif action == "graph_traverse":
        _execute_graph_traverse(state, client)
    elif action == "similarity_search":
        _execute_similarity_search(question, state, client)
    elif action == "aggregate":
        _execute_aggregate(question, state, client)
    elif action == "temporal":
        _execute_temporal(question, state, client)
    elif action == "document_retrieve":
        _execute_similarity_search(question, state, client)


# ─── Main Agentic Loop ─────────────────────────────────────────────────────

def run_investigation(question: str, client, is_agentic: bool = True) -> Dict:
    """Run a full investigation with plan-execute-evaluate-replan loop."""
    state = InvestigationState(
        question=question,
        original_question=question,
        max_steps=MAX_STEPS,
    )

    q_type = classify_question(question)
    print(f"  [Orchestrator] Question type: {q_type}")

    if is_agentic:
        plan = plan_investigation(question, state)
        state.plan = plan
        print(f"  [Orchestrator] Initial plan: {plan}")

        executed_actions = set()
        action_sequence = []

        if q_type == "temporal":
            action_sequence = ["entity_link", "temporal", "similarity_search", "synthesize_answer"]
        elif q_type == "aggregation":
            action_sequence = ["entity_link", "similarity_search", "aggregate", "synthesize_answer"]
        elif q_type == "superlative":
            action_sequence = ["entity_link", "similarity_search", "aggregate", "synthesize_answer"]
        elif q_type == "lookup":
            action_sequence = ["entity_link", "similarity_search", "aggregate", "synthesize_answer"]
        elif q_type == "venue_date":
            action_sequence = ["entity_link", "similarity_search", "temporal", "synthesize_answer"]
        else:
            action_sequence = ["entity_link", "graph_traverse", "similarity_search", "synthesize_answer"]

        for step_num in range(state.max_steps):
            state.step_count = step_num

            if step_num < len(action_sequence):
                next_action = action_sequence[step_num]
                is_sufficient = False
            else:
                evaluation = evaluate_evidence(question, state)
                is_sufficient = evaluation.get("sufficient", False)
                next_action = evaluation.get("next_action", "synthesize_answer")

            print(f"  [Step {step_num}] action={next_action}, sufficient={is_sufficient}")

            if is_sufficient or next_action == "synthesize_answer" or step_num >= state.max_steps - 1:
                break

            execute_action(next_action, question, state, client)
            executed_actions.add(next_action)

            if state.has_sufficient_evidence():
                print(f"  [Step {step_num}] Sufficient evidence collected")
                break

    else:
        print(f"  [Orchestrator] Running non-agentic mode")
        _execute_entity_link(question, state, client)
        _execute_graph_traverse(state, client)
        _execute_similarity_search(question, state, client)

        if q_type == "temporal":
            _execute_temporal(question, state, client)
        elif q_type in ("aggregation", "superlative", "lookup"):
            _execute_aggregate(question, state, client)

    all_evidence = []
    total_len = 0
    for ev in state.evidence:
        chunk = ev.content[:2000]
        if total_len + len(chunk) < 8000:
            all_evidence.append(chunk)
            total_len += len(chunk)

    state.record_action(Action.SYNTHESIZE_ANSWER, {"evidence_count": len(all_evidence)})
    final_answer = synthesize_answer(question, all_evidence, state)
    state.final_answer = final_answer
    state.is_complete = True

    return {
        "question": state.original_question,
        "answer": final_answer,
        "evidence_count": len(state.evidence),
        "steps": state.step_count,
        "tokens": state.get_total_tokens(),
        "token_usage": state.token_usage,
        "actions": state.actions_taken,
        "visited_docs": len(state.visited_doc_ids),
    }
