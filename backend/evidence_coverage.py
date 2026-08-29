"""Structured evidence coverage checks for financial answerability."""

from __future__ import annotations

import os
import re
from typing import Any

from calculation_service import resolve_frame_operands
from query_parser import (
    FIELD_ALIASES,
    STATEMENT_TYPE_LABELS,
    assess_required_field_coverage,
    infer_page_statement_types,
    matches_company_text,
)


STRUCTURED_TASKS = {"calculation", "comparison", "selection"}
_FINANCIAL_NUMBER = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_QUARTER_VARIANTS = {
    "q1": ("q1", "first quarter"),
    "q2": ("q2", "second quarter"),
    "q3": ("q3", "third quarter"),
    "q4": ("q4", "fourth quarter"),
}
_CONCEPT_STOPWORDS = {
    "a", "an", "and", "answer", "are", "as", "at", "between", "by", "calculate",
    "change", "during", "fiscal", "for", "from", "fy", "give", "in", "is", "market",
    "of", "on", "question", "round", "the", "to", "using", "was", "were", "what",
    "which", "who", "with", "year", "years", "year-over-year",
}


def stage_aware_coverage_enabled() -> bool:
    return os.getenv("STAGE_AWARE_COVERAGE_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def protected_page_slots_enabled() -> bool:
    return os.getenv("PROTECTED_PAGE_SLOTS_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def explicit_formula_advisory_enabled() -> bool:
    return os.getenv("EXPLICIT_FORMULA_ADVISORY_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _explicit_formula_operands(task_spec: dict[str, Any]) -> list[dict[str, Any]]:
    if not explicit_formula_advisory_enabled():
        return []
    if float(task_spec.get("explicit_formula_confidence") or 0.0) < 0.8:
        return []
    return [item for item in task_spec.get("explicit_formula_operands") or [] if isinstance(item, dict)]


def _coverage_fields(task_spec: dict[str, Any]) -> list[str]:
    fields = [str(value) for value in task_spec.get("required_fields") or []]
    fields.extend(str(item.get("field") or "") for item in _explicit_formula_operands(task_spec))
    return list(dict.fromkeys(field for field in fields if field))


def _coverage_periods(task_spec: dict[str, Any]) -> list[str]:
    periods = [str(value) for value in task_spec.get("required_periods") or []]
    periods.extend(str(value) for value in task_spec.get("explicit_formula_periods") or [])
    return list(dict.fromkeys(value for value in periods if value))


def _document_text(document: dict[str, Any]) -> str:
    return "\n".join(
        filter(None, [
            str(document.get("filename") or ""),
            str(document.get("doc_name") or ""),
            str(document.get("text") or document.get("page_text") or ""),
        ])
    )


def _contains_period(text: str, period: str) -> bool:
    lowered = str(text or "").casefold()
    normalized = str(period or "").casefold().strip()
    variants = _QUARTER_VARIANTS.get(normalized)
    if variants:
        return any(re.search(rf"\b{re.escape(value)}\b", lowered) for value in variants)
    if re.fullmatch(r"(?:19|20)\d{2}", normalized):
        return bool(re.search(
            rf"\b(?:fy\s*|fiscal(?:\s+year)?\s+)?{re.escape(normalized)}\b",
            lowered,
        ))
    return bool(normalized and re.search(rf"\b{re.escape(normalized)}\b", lowered))


def _concept_specs(task_spec: dict[str, Any]) -> list[tuple[str, list[str]]]:
    specs: list[tuple[str, list[str]]] = []
    fields = _coverage_fields(task_spec)
    for field in fields:
        aliases = [str(value).casefold() for value in FIELD_ALIASES.get(str(field), []) if str(value).strip()]
        specs.append((str(field), aliases or [str(field).replace("_", " ").casefold()]))
    if not fields:
        for index, concept in enumerate(task_spec.get("required_concepts") or []):
            normalized = str(concept or "").casefold().strip()
            if normalized and not any(normalized in aliases for _, aliases in specs):
                specs.append((f"concept_{index}", [normalized]))
    target = str(task_spec.get("target_measure") or "").casefold().strip()
    if (
        task_spec.get("target_measure_explicit")
        and target
        and (not fields or str(task_spec.get("task_type") or "") == "selection")
        and not any(target in aliases for _, aliases in specs)
    ):
        specs.append(("target_measure", [target]))
    return specs


def _normalized_concept_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
        if raw in _CONCEPT_STOPWORDS or re.fullmatch(r"(?:19|20)\d{2}", raw):
            continue
        token = raw
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) >= 2:
            tokens.add(token)
    return tokens


def _matches_concept(text: str, aliases: list[str]) -> tuple[bool, float, str]:
    lowered = str(text or "").casefold()
    for alias in aliases:
        normalized = str(alias or "").casefold().strip()
        if normalized and normalized in lowered:
            return True, 1.0, "phrase"
    text_tokens = _normalized_concept_tokens(lowered)
    best_score = 0.0
    for alias in aliases:
        concept_tokens = _normalized_concept_tokens(alias)
        if not concept_tokens:
            continue
        overlap = len(concept_tokens & text_tokens)
        score = overlap / len(concept_tokens)
        best_score = max(best_score, score)
        # This fallback is diagnostic only. Requiring two content words avoids
        # treating a lone generic word as evidence for a long question phrase.
        if overlap >= 2 and score >= 0.4:
            return True, score, "token_overlap"
    return False, best_score, "none"


def _page_key(document: dict[str, Any]) -> str:
    filename = str(document.get("filename") or document.get("doc_name") or "")
    return f"{filename}#page={document.get('page_number')}"


def _requirement_pages(
    task_spec: dict[str, Any],
    documents: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    requirement_pages: dict[str, list[str]] = {}
    labels: dict[str, str] = {}

    def record(requirement: str, label: str, document: dict[str, Any]) -> None:
        key = _page_key(document)
        requirement_pages.setdefault(requirement, [])
        if key not in requirement_pages[requirement]:
            requirement_pages[requirement].append(key)
        labels[requirement] = label

    concepts = _concept_specs(task_spec)
    required_periods = _coverage_periods(task_spec)
    quarter_qualifiers = [value for value in required_periods if value.casefold() in _QUARTER_VARIANTS]
    fields = _coverage_fields(task_spec)
    explicit_operands = _explicit_formula_operands(task_spec)
    explicit_by_field = {
        str(item.get("field")): item
        for item in explicit_operands
        if item.get("field")
    }
    expected_statements = {str(value) for value in task_spec.get("statement_types") or []}
    scope = str(task_spec.get("scope") or "").replace("_", " ").casefold().strip()

    for document in documents:
        text = _document_text(document)
        lowered = text.casefold()
        page_statements = set(infer_page_statement_types(text))
        concept_hits: set[str] = set()
        for concept_key, aliases in concepts:
            concept_matched, _, _ = _matches_concept(lowered, aliases)
            if concept_matched:
                requirement = f"concept:{concept_key}"
                record(requirement, aliases[0], document)
                concept_hits.add(concept_key)
        matched_fields = assess_required_field_coverage(
            {**task_spec, "company": "", "required_fields": fields, "required_periods": []},
            [document],
        ).get("matched_fields") or {}
        for field in fields:
            if field in matched_fields:
                record(f"field:{field}", field, document)
                operand = explicit_by_field.get(field)
                operand_periods = [str(value) for value in (operand or {}).get("periods") or []]
                if str(task_spec.get("task_type") or "") == "calculation" and not operand:
                    record(f"operand:{field}", field, document)
                elif operand and all(_contains_period(text, period) for period in operand_periods):
                    record(f"operand:{operand.get('key') or field}", field, document)
        for operand in explicit_operands:
            if operand.get("field"):
                continue
            aliases = [str(value) for value in operand.get("aliases") or [operand.get("label") or ""]]
            matched, _, _ = _matches_concept(text, aliases)
            operand_periods = [str(value) for value in operand.get("periods") or []]
            if matched and _FINANCIAL_NUMBER.search(text) and all(
                _contains_period(text, period) for period in operand_periods
            ):
                record(
                    f"operand:{operand.get('key')}",
                    str(operand.get("label") or operand.get("key") or ""),
                    document,
                )
        for period in required_periods:
            if _contains_period(text, period):
                record(f"period:{period}", period, document)
        if scope and re.search(rf"\b{re.escape(scope)}\b", lowered):
            record("scope", scope, document)
        if expected_statements and page_statements & expected_statements:
            for statement in sorted(expected_statements & page_statements):
                record(f"statement:{statement}", statement, document)

        task_type = str(task_spec.get("task_type") or "")
        if task_type == "comparison":
            side_periods = [value for value in required_periods if value.casefold() not in _QUARTER_VARIANTS]
            has_target_concept = bool(concept_hits or matched_fields)
            qualifiers_present = all(_contains_period(text, value) for value in quarter_qualifiers)
            for period in side_periods:
                if has_target_concept and qualifiers_present and _contains_period(text, period):
                    record(f"comparison_side:{period}", period, document)
        elif task_type == "selection" and task_spec.get("target_measure_explicit"):
            target = str(task_spec.get("target_measure") or "").casefold().strip()
            if target and target in lowered and len(_FINANCIAL_NUMBER.findall(text)) >= 2:
                record("selection_candidate_table", target, document)
    return requirement_pages, labels


def assess_stage_coverage(
    task_spec: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    """Assess QuerySpec evidence requirements at one retrieval/context stage without gold labels."""
    documents = list(documents or [])
    fields = _coverage_fields(task_spec)
    periods = _coverage_periods(task_spec)
    base = assess_required_field_coverage(
        {**task_spec, "required_fields": fields, "required_periods": periods},
        documents,
    )
    requirement_pages, requirement_labels = _requirement_pages(task_spec, documents)
    required: list[str] = []
    concepts = _concept_specs(task_spec)
    required.extend(f"concept:{key}" for key, _ in concepts)
    required.extend(f"field:{field}" for field in fields)
    required.extend(f"period:{period}" for period in periods)
    if task_spec.get("scope"):
        required.append("scope")
    task_type = str(task_spec.get("task_type") or "")
    explicit_operands = _explicit_formula_operands(task_spec)
    if explicit_operands:
        required.extend(f"operand:{item.get('key')}" for item in explicit_operands if item.get("key"))
    elif task_type == "calculation":
        required.extend(f"operand:{field}" for field in fields)
    elif task_type == "comparison":
        required.extend(
            f"comparison_side:{period}"
            for period in periods
            if period.casefold() not in _QUARTER_VARIANTS
        )
    elif task_type == "selection" and task_spec.get("target_measure_explicit"):
        required.append("selection_candidate_table")
    required = list(dict.fromkeys(required))
    satisfied = [item for item in required if requirement_pages.get(item)]
    missing = [item for item in required if item not in satisfied]

    company = str(task_spec.get("company") or "")
    company_confident = bool(company and float(task_spec.get("company_confidence") or 0.0) >= 0.8)
    normalized_company = re.sub(r"[^a-z0-9]+", " ", company.casefold()).strip()
    company_documents = []
    for document in documents:
        identity = "\n".join([str(document.get("filename") or ""), str(document.get("doc_name") or "")])
        normalized_identity = re.sub(r"[^a-z0-9]+", " ", identity.casefold()).strip()
        if company and (
            matches_company_text(identity, company)
            or (normalized_company and normalized_company in normalized_identity)
        ):
            company_documents.append(document)
    target_document_status = (
        "matched" if company_documents
        else "not_matched" if company_confident
        else "not_determined"
    )
    if not documents:
        status = "insufficient"
    elif not missing:
        status = "complete"
    elif satisfied:
        status = "partial"
    else:
        status = "insufficient"
    if stage == "candidate":
        if target_document_status == "not_matched":
            miss_diagnosis = "target_document_not_hit"
        elif status != "complete" and target_document_status == "matched":
            miss_diagnosis = "target_document_hit_requirement_page_not_hit"
        elif status == "complete":
            miss_diagnosis = "candidate_requirements_satisfied"
        else:
            miss_diagnosis = "target_document_not_determined"
    else:
        miss_diagnosis = ""
    return {
        "stage": stage,
        "coverage_basis": "query_spec_runtime_no_gold",
        "explicit_formula_advisory_enabled": explicit_formula_advisory_enabled(),
        "explicit_formula_source": task_spec.get("explicit_formula_source") or "none",
        "explicit_formula_fields": [
            str(item.get("field") or item.get("key") or "") for item in explicit_operands
        ],
        "requirements_defined": bool(required),
        "diagnostic_confidence": "field_bound" if fields else "lexical_concept_only",
        "status": status,
        "document_count": len(documents),
        "page_count": len({_page_key(document) for document in documents}),
        "required_concepts": [label for _, aliases in concepts for label in aliases[:1]],
        "missing_concepts": [
            requirement_labels.get(item, item.split(":", 1)[-1])
            for item in missing if item.startswith("concept:")
        ],
        "required_fields": fields,
        "missing_fields": [item.split(":", 1)[1] for item in missing if item.startswith("field:")],
        "required_periods": periods,
        "missing_periods": [item.split(":", 1)[1] for item in missing if item.startswith("period:")],
        "scope_status": "missing" if "scope" in missing else base.get("scope_status", "not_required"),
        "required_comparison_sides": [item.split(":", 1)[1] for item in required if item.startswith("comparison_side:")],
        "missing_comparison_sides": [item.split(":", 1)[1] for item in missing if item.startswith("comparison_side:")],
        "required_operands": (
            [str(item.get("key") or "") for item in explicit_operands]
            if explicit_operands else fields if task_type == "calculation" else []
        ),
        "missing_operands": [item.split(":", 1)[1] for item in missing if item.startswith("operand:")],
        "satisfied_requirements": satisfied,
        "missing_requirements": missing,
        "requirement_pages": requirement_pages,
        "target_document_status": target_document_status,
        "candidate_miss_diagnosis": miss_diagnosis,
    }


def coverage_transition_reason(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_satisfied = set(before.get("satisfied_requirements") or [])
    after_satisfied = set(after.get("satisfied_requirements") or [])
    if before.get("status") == "complete" and after.get("status") != "complete":
        return "coverage_lost"
    if after_satisfied > before_satisfied:
        return "coverage_improved"
    if before_satisfied == after_satisfied:
        return "coverage_preserved"
    if before_satisfied - after_satisfied:
        return "coverage_partially_lost"
    return "coverage_changed"


def _group_page_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for rank, document in enumerate(documents or []):
        key = _page_key(document)
        if key not in grouped:
            grouped[key] = {"key": key, "rank": rank, "documents": [], "texts": []}
        grouped[key]["documents"].append(document)
        text = str(document.get("text") or document.get("page_text") or "").strip()
        if text and text not in grouped[key]["texts"]:
            grouped[key]["texts"].append(text)
    pages = []
    for item in grouped.values():
        representative = dict(item["documents"][0])
        representative["text"] = "\n".join(item["texts"])
        pages.append({**item, "representative": representative})
    return pages


def protect_selected_page_slots(
    task_spec: dict[str, Any],
    candidate_documents: list[dict[str, Any]],
    selected_documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace ordinary selected pages with requirement-complete candidate pages without growing the page budget."""
    enabled = protected_page_slots_enabled()
    candidate_coverage = assess_stage_coverage(task_spec, candidate_documents, stage="candidate")
    before = assess_stage_coverage(task_spec, selected_documents, stage="selected_page_before_protection")
    trace: dict[str, Any] = {
        "protected_page_slots_enabled": enabled,
        "protected_pages": [],
        "protected_page_replacements": [],
        "protected_page_count": 0,
        "selected_page_count_before": before["page_count"],
        "selected_page_count_after": before["page_count"],
        "coverage_before": before,
        "coverage_after": before,
        "coverage_transition_reason": "disabled" if not enabled else "no_recoverable_requirement",
    }
    if not enabled or not candidate_documents or not selected_documents:
        return selected_documents, trace

    recoverable = [
        item for item in candidate_coverage.get("satisfied_requirements") or []
        if item in set(before.get("missing_requirements") or [])
    ]
    task_type = str(task_spec.get("task_type") or "")
    explicit_operand_requirements = [item for item in recoverable if item.startswith("operand:")]
    if explicit_operand_requirements:
        prioritized = explicit_operand_requirements
    elif task_type == "calculation":
        prioritized = [item for item in recoverable if item.startswith("operand:")]
    elif task_type == "comparison":
        prioritized = [item for item in recoverable if item.startswith("comparison_side:")]
        prioritized += [item for item in recoverable if item.startswith("period:") and item not in prioritized]
    elif task_type == "selection" and task_spec.get("target_measure_explicit"):
        prioritized = [item for item in recoverable if item == "selection_candidate_table"]
    elif task_type == "lookup" and task_spec.get("target_measure_explicit"):
        prioritized = [item for item in recoverable if item == "concept:target_measure"][:1]
    else:
        prioritized = []
    if task_type == "calculation":
        prioritized += [
            item for item in recoverable
            if item.startswith("field:") and item not in prioritized
        ]
    elif task_type not in {"lookup", "selection"} or task_spec.get("target_measure_explicit"):
        prioritized += [
            item for item in recoverable
            if item.startswith(("field:", "concept:")) and item not in prioritized
        ]
    if task_type == "lookup":
        prioritized = prioritized[:1]
    if not prioritized:
        return selected_documents, trace

    candidate_pages = {item["key"]: item for item in _group_page_documents(candidate_documents)}
    selected_pages = _group_page_documents(selected_documents)
    selected_keys = {item["key"] for item in selected_pages}
    protected_keys: set[str] = set()
    current_coverage = before
    for requirement in prioritized:
        if requirement in set(current_coverage.get("satisfied_requirements") or []):
            continue
        page_keys = [
            key for key in candidate_coverage.get("requirement_pages", {}).get(requirement, [])
            if key not in selected_keys and key in candidate_pages
        ]
        best_choice = None
        for candidate_key in page_keys:
            candidate_page = candidate_pages[candidate_key]
            for victim_index, victim in enumerate(selected_pages):
                if victim["key"] in protected_keys:
                    continue
                simulated = [page["representative"] for index, page in enumerate(selected_pages) if index != victim_index]
                simulated.append(candidate_page["representative"])
                after = assess_stage_coverage(task_spec, simulated, stage="selected_page_after_protection")
                before_satisfied = set(current_coverage.get("satisfied_requirements") or [])
                after_satisfied = set(after.get("satisfied_requirements") or [])
                if requirement not in after_satisfied or not before_satisfied <= after_satisfied:
                    continue
                gain = len(after_satisfied - before_satisfied)
                retrieval_score = float(candidate_page["representative"].get("rerank_score") or candidate_page["representative"].get("score") or 0.0)
                victim_value = len(
                    set(assess_stage_coverage(
                        task_spec, [victim["representative"]], stage="page_value",
                    ).get("satisfied_requirements") or [])
                )
                choice = (gain, retrieval_score, -victim_value, victim["rank"], victim_index, candidate_page, after)
                if best_choice is None or choice[:4] > best_choice[:4]:
                    best_choice = choice
        if best_choice is None:
            continue
        _, _, _, _, victim_index, candidate_page, after = best_choice
        victim = selected_pages[victim_index]
        selected_keys.remove(victim["key"])
        selected_keys.add(candidate_page["key"])
        selected_pages[victim_index] = candidate_page
        protected_keys.add(candidate_page["key"])
        trace["protected_pages"].append({
            "filename": candidate_page["representative"].get("filename"),
            "page_number": candidate_page["representative"].get("page_number"),
            "protected_page_reason": requirement,
        })
        trace["protected_page_replacements"].append({
            "protected_page": candidate_page["key"],
            "protected_page_reason": requirement,
            "replaced_page": victim["key"],
        })
        current_coverage = after

    if not trace["protected_pages"]:
        return selected_documents, trace
    output: list[dict[str, Any]] = []
    for page in selected_pages:
        if page["key"] in protected_keys:
            output.append(page["representative"])
        else:
            output.extend(page["documents"])
    trace.update({
        "protected_page_count": len(trace["protected_pages"]),
        "selected_page_count_after": len(selected_pages),
        "coverage_after": current_coverage,
        "coverage_transition_reason": coverage_transition_reason(before, current_coverage),
    })
    return output, trace


def structured_coverage_enabled() -> bool:
    return os.getenv("STRUCTURED_COVERAGE_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def structured_coverage_advisory_enabled() -> bool:
    return os.getenv("STRUCTURED_COVERAGE_ADVISORY_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _known_compatible(frames: list[dict[str, Any]], fields: tuple[str, ...]) -> bool:
    for field in fields:
        values = [str(frame.get(field)) for frame in frames if frame.get(field) not in (None, "")]
        if len(values) != len(frames) or len(set(values)) != 1:
            return False
    return bool(frames)


def _frame_concepts(task_spec: dict[str, Any]) -> list[tuple[str, list[str]]]:
    concepts: list[tuple[str, list[str]]] = []
    for field in task_spec.get("required_fields") or []:
        aliases = [str(item) for item in FIELD_ALIASES.get(str(field), [])]
        concepts.append((str(field), aliases or [str(field).replace("_", " ")]))
    target = str(task_spec.get("target_measure") or "").strip()
    task_type = str(task_spec.get("task_type") or "lookup")
    if target and (not concepts or task_type in {"comparison", "selection"}):
        existing = {item.casefold() for _, values in concepts for item in values}
        if target.casefold() not in existing:
            concepts.append(("target_measure", [target]))
    return concepts


def _frames_by_required_field(
    task_spec: dict[str, Any],
    frames: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    # Reuse the same layered resolution as the executor, but do not require
    # formula-level uniqueness for comparison/selection coverage.
    from calculation_service import match_evidence_frames_detailed  # local import avoids an import cycle

    company = str(task_spec.get("company") or "").casefold().strip()
    result: dict[str, list[dict[str, Any]]] = {}
    traces: list[dict[str, Any]] = []
    for field, concepts in _frame_concepts(task_spec):
        candidates, match_trace = match_evidence_frames_detailed(
            field,
            frames,
            concepts=concepts,
            statement_types=[str(item) for item in task_spec.get("statement_types") or []],
            scope=str(task_spec.get("scope") or ""),
        )
        if company:
            candidates = [
                frame for frame in candidates
                if str(frame.get("company") or "").casefold().strip() == company
            ]
        result[str(field)] = candidates
        traces.append(match_trace)
    return result, traces


def assess_structured_coverage(
    task_spec: dict[str, Any],
    documents: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    base_coverage: dict[str, Any],
) -> dict[str, Any]:
    """Add explicit support dimensions and a conservative answerable status."""
    task_type = str(task_spec.get("task_type") or "lookup")
    page_supported = bool(documents)
    base = {
        **base_coverage,
        "base_status": base_coverage.get("status"),
        "base_answerable": base_coverage.get("status") == "complete" and page_supported,
        "structured_coverage_enabled": True,
        "page_supported": page_supported,
        "row_supported": None,
        "period_supported": None,
        "unit_scale_supported": None,
        "scope_supported": None,
        "operands_validated": None,
        "structured_answerable": False,
        "structured_execution_ready": False,
        "structured_status": "not_applicable",
        "answerable": base_coverage.get("status") == "complete" and page_supported,
        "structured_missing": [],
    }
    # Textual lookups retain the existing coverage semantics.
    if task_type not in STRUCTURED_TASKS:
        base["coverage_basis"] = "text_lookup"
        return base

    required_fields = [str(field) for field in task_spec.get("required_fields") or []]
    field_frames, match_traces = _frames_by_required_field(task_spec, frames)
    concept_keys = [field for field, _ in _frame_concepts(task_spec)]
    row_supported = bool(concept_keys) and all(field_frames.get(field) for field in concept_keys)
    relevant_frames = list({
        str(frame.get("evidence_id") or id(frame)): frame
        for field in concept_keys
        for frame in field_frames.get(field, [])
    }.values())
    required_periods = [str(period) for period in task_spec.get("required_periods") or []]
    if required_periods:
        period_supported = row_supported and all(
            all(any(str(frame.get("period") or "") == period for frame in field_frames.get(field, [])) for period in required_periods)
            for field in concept_keys
        )
    else:
        period_supported = row_supported and bool(relevant_frames) and all(frame.get("period") for frame in relevant_frames)

    unit_scale_supported = row_supported and _known_compatible(relevant_frames, ("currency", "scale"))
    scope_supported = row_supported and _known_compatible(relevant_frames, ("scope",))
    operands_validated = False
    if task_type == "calculation" and task_spec.get("formula"):
        operands_validated = bool(resolve_frame_operands(task_spec, frames)) and unit_scale_supported and scope_supported
    elif task_type == "comparison":
        from calculation_service import resolve_comparison_frames
        operands_validated = bool(resolve_comparison_frames(task_spec, frames))
    elif task_type == "selection":
        from calculation_service import resolve_selection_frames
        operands_validated = bool(resolve_selection_frames(task_spec, frames))

    operation = str(task_spec.get("operation") or "")
    operation_confident = float(task_spec.get("operation_confidence", 1.0) or 0.0) >= 0.8
    if task_type == "calculation":
        operation_validated = operation_confident and operation not in {"", "select"}
    elif task_type == "comparison":
        operation_validated = operation_confident and operation in {"compare", "percentage_change", "subtract"}
    else:
        operation_validated = (
            operation_confident
            and operation in {"argmax", "argmin"}
            and bool(task_spec.get("candidate_dimension"))
            and bool(task_spec.get("target_measure"))
        )
    metadata_validated = bool(unit_scale_supported and scope_supported)
    period_semantics_validated = (
        "period_semantics_confidence" not in task_spec
        or len(required_periods) < 2
        or float(task_spec.get("period_semantics_confidence") or 0.0) >= 0.8
    )
    structured_gate_trace = {
        "frame_matched": bool(relevant_frames),
        "measure_validated": bool(row_supported),
        "period_validated": bool(period_supported and period_semantics_validated),
        "metadata_validated": metadata_validated,
        "operand_unique": bool(operands_validated),
        "operation_validated": operation_validated,
    }
    execution_ready = page_supported and all(structured_gate_trace.values())

    missing = []
    dimensions = {
        "page_supported": page_supported,
        "row_supported": row_supported,
        "period_supported": period_supported,
        "unit_scale_supported": unit_scale_supported,
        "scope_supported": scope_supported,
        "operands_validated": operands_validated,
    }
    for name, supported in dimensions.items():
        if supported is False:
            missing.append(name)
    structured_answerable = page_supported and operands_validated
    structured_status = "complete" if structured_answerable else "partial" if page_supported and (row_supported or bool(frames)) else "insufficient"
    base_answerable = bool(base["base_answerable"])
    advisory = structured_coverage_advisory_enabled()
    answerable = (base_answerable or structured_answerable) if advisory else structured_answerable
    if advisory:
        status = "complete" if answerable else str(base_coverage.get("status") or structured_status)
    else:
        status = structured_status
    failure_reason = ""
    if not relevant_frames:
        failure_reason = "no_related_frames"
    elif not row_supported:
        failure_reason = "missing_required_concept"
    elif not period_supported:
        failure_reason = "period_not_resolved"
    elif not period_semantics_validated:
        failure_reason = "period_order_ambiguous"
    elif not unit_scale_supported:
        failure_reason = "unit_or_scale_not_validated"
    elif not scope_supported:
        failure_reason = "scope_not_validated"
    elif not operands_validated:
        failure_reason = "operands_not_unique_or_complete"
    elif not operation_validated:
        failure_reason = "operation_not_validated"
    frame_candidates = [candidate for trace in match_traces for candidate in trace.get("candidates") or []]
    best_candidate = max(frame_candidates, key=lambda item: float(item.get("match_score") or 0), default={})
    return {
        **base,
        **dimensions,
        "answerable": answerable,
        "status": status,
        "structured_status": structured_status,
        "structured_answerable": structured_answerable,
        "structured_execution_ready": execution_ready,
        "structured_gate_trace": structured_gate_trace,
        "structured_advisory_mode": advisory,
        "structured_missing": missing,
        "coverage_basis": "evidence_frame",
        "structured_relevant_frame_count": len(relevant_frames),
        "relevant_frame_count": len(relevant_frames),
        "queryspec_concepts": [concept for _, values in _frame_concepts(task_spec) for concept in values],
        "frame_match_candidates": frame_candidates,
        "frame_match_method": best_candidate.get("match_method") or "",
        "frame_match_score": best_candidate.get("match_score") or 0.0,
        "operand_resolution_failure_reason": failure_reason,
    }


def build_document_scoped_supplemental_query(
    question: str,
    task_spec: dict[str, Any],
    coverage: dict[str, Any],
) -> str:
    """Build one deterministic query from missing concepts and constraints."""
    missing_fields = [str(item) for item in coverage.get("missing_fields") or []]
    missing_operands = [str(item) for item in coverage.get("missing_operands") or []]
    structured_missing = set(coverage.get("structured_missing") or [])
    if not missing_fields and structured_missing & {
        "row_supported", "period_supported", "unit_scale_supported", "scope_supported", "operands_validated",
    }:
        missing_fields = [str(item) for item in task_spec.get("required_fields") or []]
    labels = [FIELD_ALIASES[field][0] for field in missing_fields if FIELD_ALIASES.get(field)]
    explicit_operands = {
        str(item.get("key") or ""): item
        for item in _explicit_formula_operands(task_spec)
        if item.get("key")
    }
    operand_labels = []
    operand_periods = []
    for key in missing_operands:
        operand = explicit_operands.get(key) or {}
        label = str(operand.get("label") or key.replace("_", " ")).strip()
        if label:
            operand_labels.append(label)
        operand_periods.extend(str(value) for value in operand.get("periods") or [] if str(value))
    # A formula-operand gap is already narrowly defined by the question. Do
    # not dilute it with concepts/statements that were satisfied earlier or
    # that describe a different operand's statement.
    operand_focused = bool(missing_operands and operand_labels)
    concepts = [] if operand_focused else [
        str(item) for item in task_spec.get("required_concepts") or [] if str(item).strip()
    ]
    periods = list(dict.fromkeys(
        operand_periods if operand_focused else [
            *[str(item) for item in task_spec.get("required_periods") or []],
            *operand_periods,
        ]
    ))
    statements = [] if operand_focused else [
        STATEMENT_TYPE_LABELS.get(str(item), str(item))
        for item in task_spec.get("statement_types") or []
    ]
    scope = "" if operand_focused else str(task_spec.get("scope") or "").strip()
    anchors = list(dict.fromkeys([
        *([] if operand_focused else labels),
        *operand_labels,
        *concepts,
        *periods,
        *statements,
        *([scope] if scope else []),
    ]))
    if not anchors:
        return ""
    if operand_focused:
        return f"Missing financial formula operand: {'; '.join(anchors)}"
    return f"{question}\nMissing financial evidence: {'; '.join(anchors)}"
