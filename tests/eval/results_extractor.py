from __future__ import annotations

import argparse
import json
import asyncio
from collections import Counter
from copy import deepcopy
import re
import sys
from pathlib import Path
from typing import Any, TextIO

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from echo.workflow_sections import parse_workflow_sections, render_workflow_section

DEFAULT_RESULTS_PATH = Path(__file__).with_name("results.jsonl")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("hotpotqa_train.jsonl")
DEFAULT_EVAL_OUTPUT_PATH = Path(__file__).with_name("hotpotqa_eval.jsonl")
DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts") / "answer-check.yaml"
DEFAULT_EVAL_PROMPT_PATH = Path(__file__).with_name("prompts") / "answer-refine.yaml"
DEFAULT_SESSION_DIR = Path(__file__).resolve().parents[2] / "memory" / "chat_sessions"

DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "database_search",
            "description": "Retrieve evidence from the local wiki18_100w corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused retrieval query."},
                    "top_k": {"type": "integer", "description": "Number of passages to retrieve."},
                },
                "required": ["query", "top_k"],
            },
        },
    }
]
DEFAULT_PARALLEL_TOOL_CALLS = True

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^0-9a-zA-Z]+")
_HIDDEN_TOOL_RESULT = "[information hidden]"


def load_yaml_template(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Prompt template must be a mapping: {path}")
    system = payload.get("system")
    user = payload.get("user")
    if not isinstance(system, str) or not isinstance(user, str):
        raise SystemExit(f"Prompt template must contain string 'system' and 'user' fields: {path}")
    return {"system": system, "user": user}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_session_index(session_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not session_dir.exists():
        return index

    for path in sorted(session_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            session_id = str(payload.get("session_id") or "").strip()
            if session_id:
                index[session_id] = payload
    return index


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def matches_answer(prediction: str, answers: list[str]) -> bool:
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False

    for answer in answers:
        answer_norm = normalize_answer(answer)
        if not answer_norm:
            continue
        if pred_norm == answer_norm:
            return True
        if answer_norm in pred_norm or pred_norm in answer_norm:
            return True
    return False


def token_f1_score(prediction: str, reference_answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference_answer).split()
    if not pred_tokens and not reference_tokens:
        return 1.0
    if not pred_tokens or not reference_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    return max(token_f1_score(prediction, answer) for answer in answers)


def result_answers(result: dict[str, Any]) -> list[str]:
    answers = result.get("answers")
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list):
        answers = result.get("golden_answers")
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list):
        return []
    return [str(answer).strip() for answer in answers if str(answer).strip()]


def result_prediction(result: dict[str, Any]) -> str:
    prediction = result.get("prediction")
    if prediction is None:
        prediction = result.get("answer")
    return clean_answer_text(str(prediction or ""))


def clean_answer_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def render_check_prompt(template: dict[str, str], question: str, reference_answer: str, predicted_answer: str) -> list[dict[str, str]]:
    values = {
        "questtion": question,
        "question": question,
        "reference_answer": reference_answer,
        "predicted_answer": predicted_answer,
    }
    return [
        {"role": "system", "content": template["system"].format(**values)},
        {"role": "user", "content": template["user"].format(**values)},
    ]


def render_refine_prompt(template: dict[str, str], question: str, predicted_answer: str) -> list[dict[str, str]]:
    values = {
        "questtion": question,
        "question": question,
        "predicted_answer": predicted_answer,
    }
    return [
        {"role": "system", "content": template["system"].format(**values)},
        {"role": "user", "content": template["user"].format(**values)},
    ]


def parse_judge_label(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        label = str(payload.get("label") or "").strip().lower()
        if label in {"correct", "incorrect"}:
            return label

    match = re.search(r'"label"\s*:\s*"(correct|incorrect)"', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()

    lowered = text.lower()
    if re.search(r"\bincorrect\b", lowered):
        return "incorrect"
    if re.search(r"\bcorrect\b", lowered):
        return "correct"
    return None


async def judge_with_model(messages: list[dict[str, str]]) -> bool | None:
    try:
        from echo.chat.registry import build_chat_model

        model = build_chat_model()
        response = await model.generate_response(messages=messages)
    except Exception:
        return None

    content = (response.content or "").strip()
    if not content:
        return None
    label = parse_judge_label(content)
    if label is None:
        return None
    return label == "correct"


async def refine_with_model(messages: list[dict[str, str]]) -> str | None:
    try:
        from echo.chat.registry import build_chat_model

        model = build_chat_model()
        response = await model.generate_response(messages=messages)
    except Exception:
        return None

    content = (response.content or "").strip()
    if not content:
        return None
    return parse_refined_prediction(content)


async def judge_and_refine_with_model(messages: list[dict[str, str]]) -> dict[str, Any] | None:
    try:
        from echo.chat.registry import build_chat_model

        model = build_chat_model()
        response = await model.generate_response(messages=messages)
    except Exception:
        return None

    content = (response.content or "").strip()
    if not content:
        return None
    return parse_eval_judgment(content)


def parse_refined_prediction(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

    payload = parse_json_object(text)
    if payload is not None:
        for key in ("refined", "refined_answer", "answer", "prediction"):
            value = payload.get(key)
            if isinstance(value, str):
                return clean_answer_text(value)
        return None

    return clean_answer_text(text)


def parse_eval_judgment(text: str) -> dict[str, Any] | None:
    payload = parse_json_object(text)
    correct: bool | None = None
    refined = ""

    if payload is not None:
        correct = coerce_correct(payload.get("correct"))
        if correct is None:
            correct = coerce_correct(payload.get("label"))
        for key in ("refined", "refined_answer", "answer", "prediction"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                refined = clean_answer_text(value)
                break

    if correct is None:
        label = parse_judge_label(text)
        if label is not None:
            correct = label == "correct"

    if correct is None:
        return None
    return {"correct": correct, "refined": refined}


def parse_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    embedded = first_json_object(text)
    if embedded:
        candidates.insert(0, embedded)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def coerce_correct(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    lowered = str(value or "").strip().lower()
    if lowered in {"correct", "true", "yes", "1"}:
        return True
    if lowered in {"incorrect", "false", "no", "0"}:
        return False
    return None


def heuristic_eval_judgment(prediction: str, answers: list[str]) -> dict[str, Any]:
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return {"correct": False, "refined": ""}

    for answer in answers:
        answer_norm = normalize_answer(answer)
        if not answer_norm:
            continue
        if pred_norm == answer_norm or answer_norm in pred_norm or pred_norm in answer_norm:
            return {"correct": True, "refined": clean_answer_text(answer)}
    return {"correct": False, "refined": prediction}


def update_progress(progress: Any | None, stats: dict[str, Any]) -> None:
    if progress is None:
        return
    progress.set_postfix(
        **{
            key: stats[key]
            for key in ("matched", "correct", "written", "missing_session", "model_failed")
            if key in stats
        },
        refresh=True,
    )


def extract_messages(session_payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = session_payload.get("messages")
    if not isinstance(messages, list):
        return []

    exported: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
        if not role or (content is None and not has_tool_calls):
            continue

        payload: dict[str, Any] = {"role": role}
        if content is not None:
            payload["content"] = content
        if has_tool_calls:
            payload["tool_calls"] = tool_calls
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id.strip():
            payload["tool_call_id"] = tool_call_id
        name = message.get("name")
        if isinstance(name, str) and name.strip():
            payload["name"] = name
        exported.append(payload)
    return exported


def split_workflow_training_records(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one training example per assistant decision with step-local redaction."""
    examples: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []

    for message in messages:
        payload = _copy_message(message)
        if _is_training_target(payload):
            examples.append(_training_record([*_context_messages(context), _target_message(payload)]))

        context.append(_copy_message(payload))
        if payload.get("role") == "assistant" and _has_think_block(payload.get("content")):
            _redact_previous_tool_result(context, before=len(context) - 1)

    return examples


def _training_record(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "messages": messages,
        "parallel_tool_calls": DEFAULT_PARALLEL_TOOL_CALLS,
        "tools": deepcopy(DEFAULT_TOOLS),
    }


def _is_training_target(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    content = str(message.get("content") or "")
    if parse_workflow_sections(content, allow_unclosed=True):
        return True
    tool_calls = message.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls)


def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_copy_message(message) for message in messages]


def _context_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_context_message(message) for message in messages]


def _context_message(message: dict[str, Any]) -> dict[str, Any]:
    payload = _copy_message(message)
    if payload.get("role") == "assistant":
        payload["weight"] = 0
    return payload


def _target_message(message: dict[str, Any]) -> dict[str, Any]:
    payload = _copy_message(message)
    payload["weight"] = 1
    return payload


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    payload = dict(message)
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        payload["tool_calls"] = [dict(item) if isinstance(item, dict) else item for item in tool_calls]
    return payload


def _has_think_block(content: Any) -> bool:
    block = parse_workflow_sections(str(content or ""), allow_unclosed=True).get("think")
    return bool(str(block or "").strip())


def _redact_previous_tool_result(messages: list[dict[str, Any]], *, before: int) -> None:
    for index in range(before - 1, -1, -1):
        role = str(messages[index].get("role") or "").strip()
        if role == "tool":
            messages[index]["content"] = _redacted_tool_content(messages[index].get("content"))
            return
        if role == "assistant":
            return


def _redacted_tool_content(content: Any) -> str:
    tool_block = parse_workflow_sections(str(content or ""), allow_unclosed=True).get("tool")
    source = tool_block if tool_block is not None else str(content or "")
    heading = _first_nonempty_line(source)
    hidden = f"{heading}\n\n{_HIDDEN_TOOL_RESULT}" if heading else _HIDDEN_TOOL_RESULT
    return render_workflow_section("tool", hidden)


def _first_nonempty_line(value: str) -> str:
    for line in str(value or "").splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


async def extract_training_examples(
    results: list[dict[str, Any]],
    session_index: dict[str, dict[str, Any]],
    template: dict[str, str],
    output_handle: TextIO | None = None,
    progress: Any | None = None,
    concurrency: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    stats = {"results": 0, "matched": 0, "correct": 0, "written": 0, "missing_session": 0}
    worker_count = max(1, concurrency)
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=worker_count)
    lock = asyncio.Lock()
    update_progress(progress, stats)

    async def update_stat(name: str, amount: int = 1) -> None:
        async with lock:
            stats[name] += amount
            update_progress(progress, stats)

    async def finish_result() -> None:
        async with lock:
            if progress is not None:
                update_progress(progress, stats)
                progress.update(1)

    async def write_example(example: dict[str, Any]) -> None:
        async with lock:
            output.append(example)
            if output_handle is not None:
                output_handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                output_handle.flush()
            stats["written"] += 1
            update_progress(progress, stats)

    async def handle_result(result: dict[str, Any]) -> None:
        await update_stat("results")
        if str(result.get("status") or "").strip().lower() != "ok":
            return

        session_id = str(result.get("session_id") or "").strip()
        if not session_id or session_id not in session_index:
            await update_stat("missing_session")
            return

        question = str(result.get("question") or "").strip()
        answers = result.get("answers")
        if not isinstance(answers, list):
            answers = []
        answers = [str(answer) for answer in answers if str(answer).strip()]

        prediction = str(result.get("prediction") or result.get("answer") or "").strip()
        if not question or not answers or not prediction:
            return

        await update_stat("matched")
        session_payload = session_index[session_id]
        judge_messages = render_check_prompt(
            template,
            question,
            reference_answer="; ".join(answers),
            predicted_answer=prediction,
        )
        is_correct = await judge_with_model(judge_messages)
        if is_correct is None:
            is_correct = matches_answer(prediction, answers)

        if not is_correct:
            return

        await update_stat("correct")
        messages = extract_messages(session_payload)
        if not messages:
            return

        for example in split_workflow_training_records(messages):
            await write_example(example)

    async def producer() -> None:
        for result in results:
            await queue.put(result)
        for _ in range(worker_count):
            await queue.put(None)

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                await handle_result(item)
            finally:
                if item is not None:
                    await finish_result()
                queue.task_done()

    async with asyncio.TaskGroup() as group:
        group.create_task(producer())
        for _ in range(worker_count):
            group.create_task(worker())

    return output, stats


async def extract_eval_results(
    results: list[dict[str, Any]],
    template: dict[str, str],
    judge_template: dict[str, str] | None = None,
    output_handle: TextIO | None = None,
    progress: Any | None = None,
    concurrency: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if judge_template is None:
        judge_template = load_yaml_template(DEFAULT_PROMPT_PATH)

    output: list[dict[str, Any] | None] = [None] * len(results)
    stats: dict[str, Any] = {"results": 0, "matched": 0, "correct": 0, "written": 0, "model_failed": 0}
    worker_count = max(1, concurrency)
    queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue(maxsize=worker_count)
    lock = asyncio.Lock()
    update_progress(progress, stats)

    async def update_stat(name: str, amount: int = 1) -> None:
        async with lock:
            stats[name] += amount
            update_progress(progress, stats)

    async def finish_result() -> None:
        async with lock:
            if progress is not None:
                update_progress(progress, stats)
                progress.update(1)

    async def handle_result(index: int, result: dict[str, Any]) -> None:
        await update_stat("results")

        answers = result_answers(result)
        prediction = result_prediction(result)
        question = clean_answer_text(str(result.get("question") or ""))
        record = dict(result)
        record["answer"] = answers[0] if answers else ""
        record["prediction"] = prediction

        can_evaluate = (
            str(result.get("status") or "").strip().lower() == "ok"
            and bool(question)
            and bool(answers)
            and bool(prediction)
        )
        if can_evaluate:
            await update_stat("matched")
            refine_messages = render_refine_prompt(
                template,
                question,
                predicted_answer=prediction,
            )
            refined = await refine_with_model(refine_messages)
            model_failed = refined is None
            refined = clean_answer_text(prediction if refined is None else refined)

            judge_messages = render_check_prompt(
                judge_template,
                question,
                reference_answer="; ".join(answers),
                predicted_answer=refined,
            )
            is_correct = await judge_with_model(judge_messages)
            if is_correct is None:
                model_failed = True
                is_correct = matches_answer(refined, answers)

            if model_failed:
                await update_stat("model_failed")
            judgment = {"correct": is_correct, "refined": refined}
        else:
            judgment = {"correct": False, "refined": ""}

        refined_value = judgment.get("refined")
        refined = clean_answer_text(str(refined_value or ""))
        if refined_value is None and prediction:
            refined = prediction

        correct = bool(judgment.get("correct"))
        f1 = best_token_f1(refined, answers)
        record["correct"] = correct
        record["refined"] = refined
        record["f1"] = f1

        if correct:
            await update_stat("correct")

        async with lock:
            output[index] = record

    async def producer() -> None:
        for index, result in enumerate(results):
            await queue.put((index, result))
        for _ in range(worker_count):
            await queue.put(None)

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                await handle_result(*item)
            finally:
                if item is not None:
                    await finish_result()
                queue.task_done()

    async with asyncio.TaskGroup() as group:
        group.create_task(producer())
        for _ in range(worker_count):
            group.create_task(worker())

    records = [record for record in output if record is not None]
    if output_handle is not None:
        for record in records:
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_handle.flush()

    stats["written"] = len(records)
    update_progress(progress, stats)
    return records, stats


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HotpotQA eval artifacts from Echo workflow results.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train", help="train extracts correct sessions; eval writes answer scoring rows.")
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSION_DIR)
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--judge-prompt-path", type=Path, default=DEFAULT_PROMPT_PATH, help="Eval-mode prompt used after refinement to judge correctness.")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--concurrency", type=int, default=1, help="Number of results to judge concurrently.")
    args = parser.parse_args(argv)
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive.")
    if args.prompt_path is None:
        args.prompt_path = DEFAULT_EVAL_PROMPT_PATH if args.mode == "eval" else DEFAULT_PROMPT_PATH
    if args.output_path is None:
        args.output_path = DEFAULT_EVAL_OUTPUT_PATH if args.mode == "eval" else DEFAULT_OUTPUT_PATH
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if not args.results_path.exists():
        raise SystemExit(f"Results file does not exist: {args.results_path}")
    if not args.prompt_path.exists():
        raise SystemExit(f"Prompt template does not exist: {args.prompt_path}")
    if args.mode == "eval" and not args.judge_prompt_path.exists():
        raise SystemExit(f"Judge prompt template does not exist: {args.judge_prompt_path}")

    template = load_yaml_template(args.prompt_path)
    judge_template = load_yaml_template(args.judge_prompt_path) if args.mode == "eval" else None
    results = load_jsonl(args.results_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        with args.output_path.open("w", encoding="utf-8") as handle:
            progress = tqdm(total=len(results), desc="eval-extract", unit="result", file=sys.stderr)
            try:
                records, stats = asyncio.run(
                    extract_eval_results(
                        results,
                        template,
                        judge_template,
                        output_handle=handle,
                        progress=progress,
                        concurrency=args.concurrency,
                    )
                )
            finally:
                progress.close()

        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "results": stats["results"],
                    "matched": stats["matched"],
                    "correct": stats["correct"],
                    "written": stats["written"],
                    "model_failed": stats["model_failed"],
                    "average_f1": sum(float(record.get("f1") or 0.0) for record in records) / len(records)
                    if records
                    else 0.0,
                    "concurrency": args.concurrency,
                    "output_path": str(args.output_path),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 0

    session_index = load_session_index(args.sessions_dir)
    with args.output_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(total=len(results), desc="extract", unit="result", file=sys.stderr)
        try:
            _, stats = asyncio.run(
                extract_training_examples(
                    results,
                    session_index,
                    template,
                    output_handle=handle,
                    progress=progress,
                    concurrency=args.concurrency,
                )
            )
        finally:
            progress.close()

    print(
        json.dumps(
            {
                "mode": args.mode,
                "results": stats["results"],
                "matched": stats["matched"],
                "correct": stats["correct"],
                "written": stats["written"],
                "missing_session": stats["missing_session"],
                "concurrency": args.concurrency,
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
