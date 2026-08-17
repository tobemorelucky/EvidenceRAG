from typing import Any
import importlib
import json
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langsmith import Client, evaluate
from openai import OpenAI


# ============================================================
# 0. 基础配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

DATASET_NAME = "financebench_top40_100_v2"

# 如果你现在实际上传的数据集名字不同，就改这里
# DATASET_NAME = "financebench_top20_74_v2"

def _truncate_text(text: str, max_chars: int = 6000) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"
# ============================================================
# 1. 导入你的 RAG Agent
# ============================================================

backend_path = PROJECT_ROOT / "backend"
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))

chat_with_agent = importlib.import_module("agent").chat_with_agent


# ============================================================
# 2. 通用工具函数
# ============================================================

def _extract_answer(outputs: Any) -> str:
    """
    从 target_function 输出中提取最终回答。
    兼容 response / answer / output 字段。
    """
    if isinstance(outputs, dict):
        answer = outputs.get("response") or outputs.get("answer") or outputs.get("output")
        return str(answer or "").strip()

    if hasattr(outputs, "outputs") and isinstance(outputs.outputs, dict):
        answer = (
            outputs.outputs.get("response")
            or outputs.outputs.get("answer")
            or outputs.outputs.get("output")
        )
        return str(answer or "").strip()

    return ""


def _safe_json_loads(text: str) -> dict:
    """
    尽量从 judge 模型输出中解析 JSON。
    兼容 ```json ... ``` 代码块和少量额外文本。
    """
    text = (text or "").strip()

    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Cannot parse JSON from judge output: {text}")


# ============================================================
# 3. Target function：调用你的 RAG
# ============================================================

def target_function(inputs: dict) -> dict:
    """
    LangSmith 会对每条样本调用这个函数。

    inputs:
      {
        "question": "..."
      }

    outputs:
      {
        "response": "...",
        "rag_trace": {...}
      }
    """
    question = inputs["question"]

    session_id = f"financebench_eval_{uuid4().hex}"

    result = chat_with_agent(
        user_text=question,
        user_id="langsmith_financebench_eval_user",
        session_id=session_id,
    )

    if isinstance(result, dict):
        response_text = str(
            result.get("response")
            or result.get("answer")
            or result.get("output")
            or ""
        )
        rag_trace = result.get("rag_trace", {}) or {}
    else:
        response_text = str(result)
        rag_trace = {}

    return {
        "response": response_text,
        "rag_trace": rag_trace,
    }


# ============================================================
# 4. 豆包 judge client
# ============================================================

judge_client = OpenAI(
    api_key=os.getenv("DOUBAO_API_KEY"),
    base_url=os.getenv("DOUBAO_BASE_URL"),
)

JUDGE_MODEL = os.getenv("DOUBAO_MODEL")


def _call_doubao_json_judge(system_prompt: str, user_prompt: str) -> dict:
    resp = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw = resp.choices[0].message.content
    parsed = _safe_json_loads(raw)
    parsed["_raw"] = raw
    return parsed


# ============================================================
# 5. FinanceBench 正确性 evaluator
# ============================================================

def financebench_combined_evaluator(run, example):
    """
    省 token 的完整评估器：
    每条样本只调用一次豆包 judge，同时输出：
    1. answer correctness
    2. evidence groundedness
    3. error_type
    4. judge confidence
    """
    inputs = example.inputs or {}
    reference_outputs = example.outputs or {}
    metadata = example.metadata or {}
    outputs = run.outputs or {}

    question = str(inputs.get("question", "")).strip()
    rag_answer = _extract_answer(outputs)

    reference_answer = str(reference_outputs.get("answer", "")).strip()

    evidence = str(metadata.get("evidence", "")).strip()
    justification = str(metadata.get("justification", "")).strip()
    doc_name = str(metadata.get("doc_name", "")).strip()
    company = str(metadata.get("company", "")).strip()
    question_type = str(metadata.get("question_type", "")).strip()
    question_reasoning = str(metadata.get("question_reasoning", "")).strip()

    if not rag_answer:
        comment = (
            "Judge result: incorrect\n\n"
            "Error type: missing_answer\n\n"
            "Reason: RAG 系统返回了空答案。\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Evidence:\n{_truncate_text(evidence, 2000)}"
        )
        return [
            {
                "key": "fb_answer_correctness",
                "score": 0,
                "comment": comment,
            },
            {
                "key": "fb_evidence_groundedness",
                "score": 0,
                "comment": comment,
            },
            {
                "key": "fb_judge_confidence",
                "score": 1,
                "comment": "Empty RAG answer.",
            },
            {
                "key": "fb_error_type",
                "value": "missing_answer",
                "comment": "RAG returned an empty answer.",
            },
        ]

    system_prompt = """
You are a strict but fair evaluator for a financial document RAG system.

You need to evaluate ONE RAG answer using the given question, reference answer, evidence, and justification.

Your job is to return four things:
1. Whether the RAG answer is correct.
2. Whether the RAG answer is grounded in the provided evidence.
3. The main error type.
4. Your confidence.

Evaluation rules:

Answer correctness:
- Mark answer_correct = true if the RAG answer conveys the same factual meaning as the reference answer.
- For financial numbers, units, percentages, years, company names, and financial metrics, be strict.
- Minor wording differences are acceptable.
- Extra explanation is acceptable only if it does not contradict the reference answer or evidence.
- If the RAG answer gives a wrong number, wrong year, wrong company, wrong metric, or unsupported conclusion, mark answer_correct = false.
- If the RAG answer says it cannot answer even though the reference answer and evidence are available, mark answer_correct = false.
- If the RAG answer is approximate but does not change the meaning, it can be correct.
- Do not require exact string match.

Evidence groundedness:
- Mark grounded = true if the key claims in the RAG answer are supported by the provided evidence.
- Mark grounded = false if the RAG answer introduces important numbers, dates, entities, metrics, or conclusions not supported by the evidence.
- Mark grounded = false if the RAG answer contradicts the evidence.
- It is acceptable if the RAG answer is shorter than the evidence.
- It is acceptable if the RAG answer paraphrases the evidence.

Error types:
- correct: answer is correct and grounded
- wrong_number: wrong numerical value, amount, percentage, or financial figure
- wrong_entity: wrong company, segment, product, geography, or entity
- wrong_time_period: wrong fiscal year, quarter, or period
- wrong_metric: wrong financial metric or denominator
- unsupported_claim: answer contains claims not supported by evidence
- missing_answer: answer fails to provide the required answer
- irrelevant: answer is unrelated to the question
- over_answer: answer gives multiple alternatives or excessive unsupported options
- unclear: cannot judge confidently

Return JSON only, without Markdown:
{
  "answer_correct": true or false,
  "grounded": true or false,
  "error_type": "correct | wrong_number | wrong_entity | wrong_time_period | wrong_metric | unsupported_claim | missing_answer | irrelevant | over_answer | unclear",
  "confidence": 0.0 to 1.0,
  "reason": "brief but specific explanation"
}
""".strip()

    # 为了省 token，做长度限制
    evidence_short = _truncate_text(evidence, max_chars=6000)
    justification_short = _truncate_text(justification, max_chars=2000)
    rag_answer_short = _truncate_text(rag_answer, max_chars=3000)

    user_prompt = f"""
Question:
{question}

Company:
{company}

Document name:
{doc_name}

Question type:
{question_type}

Question reasoning:
{question_reasoning}

Reference answer:
{reference_answer}

Evidence:
{evidence_short}

Justification:
{justification_short}

RAG answer:
{rag_answer_short}
""".strip()

    try:
        parsed = _call_doubao_json_judge(system_prompt, user_prompt)

        answer_correct = bool(parsed.get("answer_correct"))
        grounded = bool(parsed.get("grounded"))
        error_type = str(parsed.get("error_type", "unclear")).strip()
        reason = str(parsed.get("reason", "")).strip()
        raw = str(parsed.get("_raw", "")).strip()

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        allowed_error_types = {
            "correct",
            "wrong_number",
            "wrong_entity",
            "wrong_time_period",
            "wrong_metric",
            "unsupported_claim",
            "missing_answer",
            "irrelevant",
            "over_answer",
            "unclear",
        }

        if error_type not in allowed_error_types:
            error_type = "unclear"

        detailed_comment = (
            f"Answer correct: {answer_correct}\n\n"
            f"Grounded: {grounded}\n\n"
            f"Error type: {error_type}\n\n"
            f"Confidence: {confidence}\n\n"
            f"Judge reason: {reason}\n\n"
            f"Question:\n{question}\n\n"
            f"Company:\n{company}\n\n"
            f"Doc name:\n{doc_name}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Evidence:\n{evidence_short}\n\n"
            f"Justification:\n{justification_short}\n\n"
            f"RAG answer:\n{rag_answer_short}\n\n"
            f"Raw judge output:\n{raw}"
        )

        return [
            {
                "key": "fb_answer_correctness",
                "score": 1 if answer_correct else 0,
                "comment": detailed_comment,
            },
            {
                "key": "fb_evidence_groundedness",
                "score": 1 if grounded else 0,
                "comment": detailed_comment,
            },
            {
                "key": "fb_judge_confidence",
                "score": confidence,
                "comment": reason,
            },
            {
                "key": "fb_error_type",
                "value": error_type,
                "comment": reason,
            },
        ]

    except Exception as e:
        comment = (
            f"Judge failed: {type(e).__name__}: {e}\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Evidence:\n{evidence_short}\n\n"
            f"RAG answer:\n{rag_answer_short}"
        )

        return [
            {
                "key": "fb_answer_correctness",
                "score": 0,
                "comment": comment,
            },
            {
                "key": "fb_evidence_groundedness",
                "score": 0,
                "comment": comment,
            },
            {
                "key": "fb_judge_confidence",
                "score": 0,
                "comment": comment,
            },
            {
                "key": "fb_error_type",
                "value": "judge_failed",
                "comment": comment,
            },
        ]

# ============================================================
# 6. 启动前检查
# ============================================================

def check_dataset_structure() -> None:
    """
    简单检查 LangSmith 数据集结构是否正确：
    Inputs 里应有 question
    Reference Outputs 里应有 answer
    Metadata 里应有 evidence / doc_name 等
    """
    client = Client()

    examples = list(client.list_examples(dataset_name=DATASET_NAME, limit=1))
    if not examples:
        raise RuntimeError(f"数据集 {DATASET_NAME} 中没有样本。")

    ex = examples[0]

    print("\n=== 数据集结构检查 ===")
    print("inputs:", ex.inputs)
    print("outputs:", ex.outputs)
    print("metadata keys:", list((ex.metadata or {}).keys()))
    print("metadata evidence preview:", str((ex.metadata or {}).get("evidence", ""))[:300])

    if "question" not in (ex.inputs or {}):
        raise RuntimeError("数据集 inputs 中没有 question 字段。")

    if "answer" not in (ex.outputs or {}):
        raise RuntimeError("数据集 reference outputs 中没有 answer 字段。")

    if "evidence" not in (ex.metadata or {}):
        print("警告：metadata 中没有 evidence 字段。可以继续跑，但 correctness 的 evidence 判断会变弱。")


# ============================================================
# 7. 运行评估
# ============================================================

if __name__ == "__main__":
    print("LANGSMITH_API_KEY exists:", bool(os.getenv("LANGSMITH_API_KEY")))
    print("LANGCHAIN_API_KEY exists:", bool(os.getenv("LANGCHAIN_API_KEY")))
    print("DOUBAO_API_KEY exists:", bool(os.getenv("DOUBAO_API_KEY")))
    print("DOUBAO_BASE_URL:", os.getenv("DOUBAO_BASE_URL"))
    print("DOUBAO_MODEL:", os.getenv("DOUBAO_MODEL"))
    print("DATASET_NAME:", DATASET_NAME)

    if not os.getenv("DOUBAO_API_KEY"):
        raise RuntimeError("缺少 DOUBAO_API_KEY，请检查 .env。")

    if not os.getenv("DOUBAO_BASE_URL"):
        raise RuntimeError("缺少 DOUBAO_BASE_URL，请检查 .env。")

    if not os.getenv("DOUBAO_MODEL"):
        raise RuntimeError("缺少 DOUBAO_MODEL，请检查 .env。")

    check_dataset_structure()

    evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=[
            financebench_combined_evaluator,
        ],
        experiment_prefix="FinanceBench Answer Correctness Evaluation",
        max_concurrency=1,
    )