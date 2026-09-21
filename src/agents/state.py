from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class Action(Enum):
    ENTITY_LINK = "entity_link"
    GRAPH_TRAVERSE = "graph_traverse"
    SIMILARITY_SEARCH = "similarity_search"
    DOCUMENT_RETRIEVE = "document_retrieve"
    AGGREGATE = "aggregate"
    MULTI_HOP_REASON = "multi_hop_reason"
    EVALUATE_EVIDENCE = "evaluate_evidence"
    SYNTHESIZE_ANSWER = "synthesize_answer"


@dataclass
class Evidence:
    source: str
    content: str
    doc_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0
    step_obtained: int = 0


@dataclass
class InvestigationState:
    question: str
    original_question: str
    evidence: List[Evidence] = field(default_factory=list)
    visited_doc_ids: List[str] = field(default_factory=list)
    visited_entities: List[str] = field(default_factory=list)
    sub_questions: List[str] = field(default_factory=list)
    current_sub_question: Optional[str] = None
    step_count: int = 0
    max_steps: int = 15
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    is_complete: bool = False
    final_answer: Optional[str] = None
    plan: List[str] = field(default_factory=list)

    def add_evidence(self, evidence: Evidence):
        self.evidence.append(evidence)
        for doc_id in evidence.doc_ids:
            if doc_id not in self.visited_doc_ids:
                self.visited_doc_ids.append(doc_id)

    def has_sufficient_evidence(self) -> bool:
        if not self.evidence:
            return False
        if len(self.evidence) >= 5 and self.step_count >= 3:
            return True
        if len(self.evidence) >= 2 and self.step_count >= 4:
            return True
        if self.step_count >= self.max_steps - 1:
            return True
        return False

    def record_action(self, action: Action, details: Dict[str, Any]):
        self.actions_taken.append({
            "step": self.step_count,
            "action": action.value,
            "details": details
        })
        self.step_count += 1

    def add_token_usage(self, model: str, prompt_tokens: int, completion_tokens: int):
        if model not in self.token_usage:
            self.token_usage[model] = {"prompt": 0, "completion": 0}
        self.token_usage[model]["prompt"] += prompt_tokens
        self.token_usage[model]["completion"] += completion_tokens

    def get_total_tokens(self) -> int:
        total = 0
        for model_data in self.token_usage.values():
            total += model_data["prompt"] + model_data["completion"]
        return total

    def to_context_string(self) -> str:
        lines = [f"Question: {self.question}"]
        lines.append(f"Steps taken: {self.step_count}/{self.max_steps}")
        if self.sub_questions:
            lines.append(f"Sub-questions: {self.sub_questions}")
        if self.evidence:
            lines.append("Evidence collected:")
            for i, ev in enumerate(self.evidence):
                lines.append(f"  [{i+1}] ({ev.source}) {ev.content[:200]}...")
        if self.visited_doc_ids:
            lines.append(f"Docs visited: {len(self.visited_doc_ids)}")
        return "\n".join(lines)
