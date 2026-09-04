# Finance Reasoning Prompt v1

## Scope

This experiment changes answer generation only. Retrieval, RRF, Jina, chunks, evidence selection, Evidence Assembly, and the 28000-character context budget remain frozen. `ANSWER_PROMPT_MODE` defaults to `baseline`; `finance_reasoning` must be selected explicitly.

## Answer call chain

### Jina full baseline

1. `scripts/run_jina_full_baseline_v1.py::run_question` receives the frozen RRF candidates and cached/new Jina order.
2. `scripts/jina_full_baseline_v1.py::build_context` concatenates the selected raw chunks with source/page headers under the frozen character budget.
3. The runner calls `backend/answer_generator.py::generate_answer` with profile `clean_baseline`.
4. `build_answer_messages` resolves `ANSWER_PROMPT_MODE`, constructs the system/user messages, and invokes DeepSeek-V4-Flash through the OpenAI-compatible Ark endpoint.

### Production chat

`backend/agent.py` calls `prepare_rag_response` before answer generation. If `evidence_status == "insufficient"`, it returns a fixed insufficient-evidence message without calling the LLM. Otherwise it calls `generate_answer` or `stream_answer`. This pre-generation gate is preserved.

## Existing uncertainty, refusal, confidence, and validation behavior

- Both baseline and general prompts instruct the model to report genuinely missing evidence rather than guess. This is prompt-level refusal guidance, not a numeric confidence threshold.
- Production `agent.py` contains an `evidence_status == "insufficient"` short circuit. It is not removed or relaxed.
- The Jina full-baseline experiment does not use that short circuit: any non-empty frozen context is passed directly to answer generation.
- Clean-baseline responses bypass structured consistency, numeric display, and required-facet validators in `_finalize_generated_answer`.
- Non-clean production profiles may apply deterministic structured/numeric/facet validators after generation. These validators are unchanged.
- Query parsing and evidence coverage contain confidence values and statuses upstream, but `answer_generator.py` itself has no confidence threshold.

## Finance reasoning mode

The finance prompt asks the model to classify the task internally, extract and validate calculation operands, show a concise formula, preserve signs and comparable periods/scopes, normalize PP&E/SG&A/EPS terminology, and avoid unnecessary refusal when the evidence already supplies the required facts. The policy never supplies values and does not authorize model-memory answers.

Output remains plain answer text plus the existing usage metadata tuple. Public API schemas are unchanged.

## Smoke experiment

`scripts/run_finance_reasoning_prompt_smoke.py` deterministically samples three completed questions from the frozen Jina all100 state. Each question is generated once with `baseline` and once with `finance_reasoning`, using the exact same evidence hash. It calls neither Retrieval, Jina, nor Judge and writes `question_id`, `question_type`, `prompt_mode`, and `final_answer` to `reports/finance_reasoning_prompt_v1_smoke3/results.jsonl`.
