from __future__ import annotations

import argparse
import asyncio
import json
import re
import struct
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from echo.settings import load_app_settings

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CORPUS_PATH = DEFAULT_DATA_DIR / "retrieval_corpus" / "wiki18_100w.jsonl"
DEFAULT_INDEX_PATH = DEFAULT_DATA_DIR / "retrieval_corpus" / "e5_flat_inner.index"
DEFAULT_DATABASE_SEARCH_TOP_K = 4
DATASET_PATHS = {
    "hotpotqa": DEFAULT_DATA_DIR / "hotpotqa" / "test-1000.jsonl",
    "2wiki": DEFAULT_DATA_DIR / "2wikimultihopqa" / "test-1000.jsonl",
    "2wikimultihopqa": DEFAULT_DATA_DIR / "2wikimultihopqa" / "test-1000.jsonl",
    "musique": DEFAULT_DATA_DIR / "musique" / "test-1000.jsonl",
}


def database_search_top_k_limit() -> int:
    return load_app_settings().max_database_search_top_k


def clamp_database_search_top_k(
    top_k: Any,
    default: int = DEFAULT_DATABASE_SEARCH_TOP_K,
    max_top_k: int | None = None,
) -> int:
    try:
        requested = int(top_k or default)
    except (TypeError, ValueError):
        requested = default
    limit = max_top_k if max_top_k is not None else database_search_top_k_limit()
    return max(1, min(requested, max(1, limit)))


class JsonlDocstore:
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self.offset_path = Path(str(corpus_path) + ".offsets.u64")
        ensure_offsets(corpus_path, self.offset_path)
        self._corpus = corpus_path.open("rb")
        self._offsets = self.offset_path.open("rb")

    def close(self):
        self._offsets.close()
        self._corpus.close()

    def get(self, row: int) -> dict[str, Any]:
        self._offsets.seek(row * 8)
        payload = self._offsets.read(8)
        if len(payload) != 8:
            raise IndexError(f"Corpus row {row} is missing from {self.offset_path}.")
        self._corpus.seek(struct.unpack("<Q", payload)[0])
        return json.loads(self._corpus.readline().decode("utf-8"))


class FlashRAGIndex:
    def __init__(self, *, corpus_path: Path, index_path: Path, faiss_mmap: bool):
        import faiss

        flags = faiss.IO_FLAG_MMAP if faiss_mmap and hasattr(faiss, "IO_FLAG_MMAP") else 0
        self.faiss = faiss
        self.index = faiss.read_index(str(index_path), flags)
        self.docstore = JsonlDocstore(corpus_path)
        self.lock = threading.RLock()

    def close(self):
        self.docstore.close()

    def database_search(self, query: str, top_k: int = 4) -> dict[str, Any]:
        import numpy as np

        query_vector = np.asarray([embed_query(query)], dtype="float32")
        limit = clamp_database_search_top_k(top_k)
        items = []
        with self.lock:
            self.faiss.normalize_L2(query_vector)
            _, ids = self.index.search(query_vector, limit)
            for row in ids[0]:
                row = int(row)
                if row < 0:
                    continue
                record = self.docstore.get(row)
                items.append(
                    {
                        "title": corpus_title(record, row),
                        "content": corpus_text(record),
                        "source_type": "flashrag_wikipedia_dpr",
                        "database_name": "wiki18_100w",
                    }
                )
        return {"type": "context", "skill_name": "database_search", "items": items}


class FlashRAGToolClient:
    def __init__(self, index: FlashRAGIndex):
        self.index = index

    @property
    def tool_names(self) -> set[str]:
        return {"database_search"}

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "database_search",
                    "description": "Retrieve evidence from the local wiki18_100w corpus.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Focused retrieval query."},
                            "top_k": {
                                "type": "integer",
                                "description": f"Number of passages to retrieve, capped at {database_search_top_k_limit()}.",
                                "minimum": 1,
                                "maximum": database_search_top_k_limit(),
                            },
                        },
                        "required": ["query", "top_k"],
                    },
                },
            }
        ]

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name != "database_search":
            return {"type": "context", "skill_name": name, "items": [], "error": f"Unsupported eval tool: {name}"}
        query = clean(str(args.get("query") or ""))
        return await asyncio.to_thread(self.index.database_search, query, clamp_database_search_top_k(args.get("top_k")))


def flashrag_tool_client_factory(index: FlashRAGIndex):
    @asynccontextmanager
    async def context():
        yield FlashRAGToolClient(index)

    return context


def embed_query(query: str) -> list[float]:
    from mcp_server.rag.embedder import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder.embed_query(query)


def ensure_offsets(corpus_path: Path, offset_path: Path) -> int:
    if offset_path.exists():
        return offset_path.stat().st_size // 8

    offset_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with corpus_path.open("rb") as source, offset_path.open("wb") as target:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            target.write(struct.pack("<Q", offset))
            count += 1
    return count


def iter_jsonl_with_rows(path: Path, limit: int | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if not line.strip():
                continue
            yield row, json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                return


def corpus_text(record: dict[str, Any]) -> str:
    if isinstance(record.get("contents"), str):
        return clean(record["contents"])
    title = str(record.get("title") or "").strip()
    text = str(record.get("text") or record.get("document") or record.get("passage") or "").strip()
    return clean(f"{title}\n{text}" if title and text else title or text)


def corpus_title(record: dict[str, Any], row: int) -> str:
    title = str(record.get("title") or "").strip()
    if title:
        return title
    text = corpus_text(record)
    return text.splitlines()[0].strip() if text else f"wiki18_100w #{row}"


def question(record: dict[str, Any]) -> str:
    return clean(str(record.get("question") or record.get("query") or ""))


def answers(record: dict[str, Any]) -> list[str]:
    raw = record.get("golden_answers", record.get("answers", record.get("answer", [])))
    if isinstance(raw, str):
        raw = [raw]
    return [clean(str(item)) for item in raw if clean(str(item))] if isinstance(raw, list) else []


def eval_workflow_factory(index: FlashRAGIndex):
    from echo.chat.registry import build_chat_model
    from echo.workflow.service import WorkflowService

    return lambda: WorkflowService(
        model_factory=build_chat_model,
        tool_client_factory=flashrag_tool_client_factory(index),
    )


async def workflow_answer(question_text: str, index: FlashRAGIndex, session_id: str) -> str:
    from echo.chat.service import ChatService

    service = ChatService(workflow_factory=eval_workflow_factory(index))
    service.delete_session(session_id)
    final_answer = ""
    async for event in service.stream_message(question_text, session_id=session_id):
        if event.get("event") == "done":
            final_answer = str(event.get("data", {}).get("reply") or "")
    return final_answer


def record_key(row: int, record: dict[str, Any]) -> str:
    raw_id = record.get("id") or record.get("_id")
    return str(raw_id) if raw_id else f"row:{row}"


def question_session_id(base_session_id: str, key: str) -> str:
    safe_key = re.sub(r"[^A-Za-z0-9_.:-]+", "_", key).strip("_")
    return f"{base_session_id}:{safe_key or 'question'}"


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                key = str(payload.get("id") or payload.get("key") or "").strip()
                if key:
                    results[key] = payload
    return results


def should_rerun_result(result: dict[str, Any] | None) -> bool:
    if result is None:
        return True
    return str(result.get("status") or "").strip().lower() == "error"


def is_retryable_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    retry_markers = (
        "ratelimit",
        "rate_limit",
        "rate limit",
        "quota",
        "usage limit",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection",
        "overloaded",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    return any(marker in name or marker in text for marker in retry_markers)


async def workflow_answer_with_retries(
    question_text: str,
    index: FlashRAGIndex,
    session_id: str,
    *,
    retry_attempts: int,
    retry_backoff_seconds: float,
    retry_max_sleep_seconds: float,
) -> tuple[str, int]:
    attempts = max(1, retry_attempts)
    for attempt in range(1, attempts + 1):
        try:
            return await workflow_answer(question_text, index, session_id), attempt
        except Exception as exc:
            if attempt >= attempts or not is_retryable_error(exc):
                raise
            sleep_seconds = min(retry_max_sleep_seconds, retry_backoff_seconds * (2 ** (attempt - 1)))
            await asyncio.sleep(max(0.0, sleep_seconds))
    raise RuntimeError("retry loop exited unexpectedly")


async def evaluate_async(args: argparse.Namespace) -> dict[str, Any]:
    session_id = args.session_id or f"eval-{args.dataset}"
    index = FlashRAGIndex(
        corpus_path=args.corpus_path,
        index_path=args.flashrag_index_path,
        faiss_mmap=args.faiss_mmap,
    )
    try:
        records = list(iter_jsonl_with_rows(args.dataset_path, limit=positive_or_none(args.max_questions)))
        existing_results = load_results(args.results_path)
        pending_records = [
            (row, record)
            for row, record in records
            if should_rerun_result(existing_results.get(record_key(row, record)))
        ]
        queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue(maxsize=args.concurrency)
        results = dict(existing_results)
        lock = asyncio.Lock()
        total = len(pending_records)
        progress = tqdm(total=total, desc="eval", unit="q", file=sys.stderr)

        async def evaluate_one(row: int, record: dict[str, Any]) -> dict[str, Any]:
            key = record_key(row, record)
            q = question(record)
            golds = answers(record)
            if not q or not golds:
                return {"status": "skipped", "id": key, "row": row}

            per_question_session_id = question_session_id(session_id, key)
            try:
                prediction, attempts = await workflow_answer_with_retries(
                    q,
                    index,
                    per_question_session_id,
                    retry_attempts=args.retry_attempts,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    retry_max_sleep_seconds=args.retry_max_sleep_seconds,
                )
            except Exception as exc:
                return {
                    "status": "error",
                    "id": key,
                    "row": row,
                    "question": q,
                    "answers": golds[:5],
                    "session_id": per_question_session_id,
                    "error_type": type(exc).__name__,
                    "error": clean(str(exc)),
                    "retryable": is_retryable_error(exc),
                }
            return {
                "status": "ok",
                "id": key,
                "row": row,
                "question": q,
                "answers": golds[:5],
                "session_id": per_question_session_id,
                "prediction": prediction,
                "attempts": attempts,
            }

        async def producer() -> None:
            for row, record in pending_records:
                await queue.put((row, record))
            for _ in range(args.concurrency):
                await queue.put(None)

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    result = await evaluate_one(*item)
                    async with lock:
                        results[result["id"]] = result
                        ordered_results = [
                            results[record_key(row, record)]
                            for row, record in records
                            if record_key(row, record) in results
                        ]
                        args.results_path.parent.mkdir(parents=True, exist_ok=True)
                        with args.results_path.open("w", encoding="utf-8") as handle:
                            for item in ordered_results:
                                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        progress.update(1)
                finally:
                    queue.task_done()

        async with asyncio.TaskGroup() as group:
            group.create_task(producer())
            for _ in range(args.concurrency):
                group.create_task(worker())
    finally:
        progress.close()
        index.close()

    ordered_results = [
        results[record_key(row, record)]
        for row, record in records
        if record_key(row, record) in results
    ]

    return {
        "questions": len(ordered_results),
        "session_id": session_id,
        "concurrency": args.concurrency,
        "skipped": sum(1 for item in ordered_results if item["status"] == "skipped"),
        "errors": sum(1 for item in ordered_results if item["status"] == "error"),
        "results": ordered_results,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(evaluate_async(args))


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def positive_or_none(value: int) -> int | None:
    return None if value <= 0 else value


def existing_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{label} file does not exist: {path}")
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Echo workflow answers on a FlashRAG QA dataset with the wiki18_100w index.")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--flashrag-index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset", default=None, help="Dataset name used in default session ids, e.g. hotpotqa, 2wikimultihopqa, or musique.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="JSONL dataset path with question and answer fields.")
    parser.add_argument("--hotpotqa-path", type=Path, default=None, help="Deprecated alias for --dataset-path.")
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--max-questions", type=int, default=50, help="0 means all questions.")
    parser.add_argument("--session-id", default=None, help="Base chat memory session id. Defaults to eval-<dataset>; each question appends its stable record id.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of questions to evaluate concurrently.")
    parser.add_argument("--faiss-mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--retry-backoff-seconds", type=float, default=10.0)
    parser.add_argument("--retry-max-sleep-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)

    if args.max_questions < 0:
        raise SystemExit("--max-questions cannot be negative.")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive.")
    if args.retry_attempts <= 0:
        raise SystemExit("--retry-attempts must be positive.")
    if args.retry_backoff_seconds < 0:
        raise SystemExit("--retry-backoff-seconds cannot be negative.")
    if args.retry_max_sleep_seconds < 0:
        raise SystemExit("--retry-max-sleep-seconds cannot be negative.")
    args.corpus_path = existing_file(args.corpus_path, "wiki18_100w corpus")
    args.flashrag_index_path = existing_file(args.flashrag_index_path, "FlashRAG FAISS index")
    if args.dataset_path is not None and args.hotpotqa_path is not None and args.dataset_path != args.hotpotqa_path:
        raise SystemExit("--dataset-path and --hotpotqa-path point to different files.")
    if args.dataset_path is None and args.hotpotqa_path is None:
        dataset = args.dataset or "hotpotqa"
        args.dataset_path = DATASET_PATHS.get(dataset)
        if args.dataset_path is None:
            choices = ", ".join(sorted(DATASET_PATHS))
            raise SystemExit(f"Unknown dataset '{dataset}'. Use --dataset-path or one of: {choices}.")
    else:
        args.dataset_path = args.dataset_path or args.hotpotqa_path
    args.dataset_path = existing_file(args.dataset_path, "dataset")
    if args.dataset is None:
        args.dataset = args.dataset_path.parent.name or args.dataset_path.stem
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
