# Echo RAG Eval

This directory contains the HotpotQA / FlashRAG evaluation workflow used to test Echo's RAG behavior against a fixed `wiki18_100w` corpus.

The eval path intentionally uses a narrow tool surface:

- `database_search` is the only runtime tool exposed by `tests/eval/eval.py`.
- The workflow still uses Echo's current native tool-call protocol.
- Parallel tool calls are allowed in exported training records, but this eval tool client only exposes one tool.

## Index

Use `wiki18_100w` as the database with the `E5-base-v2` embedder.

The retrieval corpus is large and must be downloaded manually. Download `wiki18_100w` from ModelScope:

```text
https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/tree/master/retrieval_corpus
```

Place the corpus and prebuilt FAISS index under:

```text
tests/data/retrieval_corpus/
```

Required files:

- `wiki18_100w.jsonl`
- `e5_flat_inner.index`

The corpus has a prebuilt FAISS index using `intfloat/e5-base-v2`. Echo expects embeddings from an OpenAI-compatible `/v1/embeddings` endpoint, so launch the local E5 service before evaluation:

```bash
python -m mcp_server.local_e5_embedder --host 127.0.0.1 --port 8101 --model intfloat/e5-base-v2
```

Then configure `models.json`:

- set the active chat model
- set the active embedding model
- point the embedding model `base_url` at `http://127.0.0.1:8101/v1`
- use `intfloat/e5-base-v2` as the embedding model name

For eval-only prompting, replace the normal search skill with `tests/eval/prompts/SKILL-eval.md` so the model only receives `database_search` guidance.

## Train Data Generation

Training data is generated from a HotpotQA train subset.

Current subset shape:

- total records: `1000`
- sampled difficulty counts: `{"easy": 400, "medium": 300, "hard": 300}`
- source availability snapshot: `{"easy": 17972, "medium": 56814, "hard": 15661}`

Training run notes:

- Use the current Echo workflow prompt format: `<echo_plan>`, `<echo_think>`, `<echo_answer>`.
- Tool calls must be provider-native tool calls, not XML text.
- `<echo_think>` should extract reusable evidence into `valid_information`.
- Previous tool evidence is hidden from later model-facing memory after the following `<echo_think>`.
- Only samples with correct final answers should be exported for training.
- Exported training examples set `parallel_tool_calls` to `true`.

Run generation:

```bash
python tests/eval/eval.py \
  --hotpotqa-path tests/data/hotpotqa/train-1000.jsonl \
  --results-path tests/eval/results_train.jsonl \
  --max-questions 0 \
  --concurrency 8
```

Extract training-ready records:
```bash
python tests/eval/results_extractor.py \
  --mode train \
  --results-path tests/eval/results_train.jsonl \
  --output-path tests/eval/hotpotqa_train.jsonl \
  --concurrency 20
```

also check trainning data quality through:
```bash
python tests/eval/results_extractor.py \
  --mode eval \
  --results-path tests/eval/results_train.jsonl \
  --output-path tests/eval/hotpotqa_train_eval.jsonl \
  --concurrency 20
```

## Test / Eval

The eval path uses a HotpotQA dev subset of 1000 samples.

Echo answers may include evidence and explanatory text, so raw generated answers do not always align cleanly with token-level F1. The eval extractor therefore supports two scoring aids:

- answer refinement before F1, to align the prediction with a short-answer metric
- LLM-as-a-judge correctness labels for an accuracy-style score

Run eval generation:

```bash
python tests/eval/eval.py \
  --dataset 2wiki \
  --results-path 2wiki_test-v2.jsonl \
  --max-questions 0 \
  --concurrency 8
```

Extract eval rows and scores:

```bash
python tests/eval/results_extractor.py \
  --mode eval \
  --results-path 2wiki_test-v2.jsonl \
  --output-path 2wiki_test_results-v2.jsonl \
  --concurrency 20
```


## Baselines And Reporting

Reference notes from earlier runs:

- auto-rag llama3-8b baseline: `44.9`
- qwen3.5-9b baseline with the system prompt

Echo fine-tuning targets:

- finetuned llama3-8b with the Echo workflow prompt
- finetuned qwen3.5-9b with the Echo workflow prompt

Report both:

- refined-answer F1
- LLM-judge accuracy

Also report:
- model/provider used
- corpus/index versions
- embedding endpoint/model
- `settings.json`
- active workflow prompt and skill files
