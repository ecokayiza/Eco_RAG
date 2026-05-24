# Echo RAG Eval

## Index

Use `wiki18_100w` as the database with the `E5-base-v2` embedder.

The retrieval corpus is large and must be downloaded manually. Download `wiki18_100w` from ModelScope:
```text
https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/tree/master/retrieval_corpus
```
we have prepared the download script, after it, put the corpus under `tests/data/retrieval_corpus`, including `wiki18_100w.jsonl` and `e5_flat_inner.index`

the corpus has prebuilt FAISS index with e5-base-v2, we prepared local embedding model serivce, you can launch it with
```bash
python -m mcp_server.local_e5_embedder --host 127.0.0.1 --port 8101 --model intfloat/e5-base-v2
```

then **replace search skill  with `SKILL-eval.md`** to allow database_search only for evaluation.

Before testing, you should ***set the active chat and embedding model*** in `models.json`. If the active embedding model points 

## Train
- - From HotpotQA train including 1000 records with easy medium hard:
- - counts={'easy': 400, 'hard': 300, 'medium': 300};
- - available={'easy': 17972,'medium': 56814, 'hard': 15661}
- use chatgpt-5.5
- replace system prompt with `system-train.yaml` , for `<echo_think>`, include validation deciding if the previous tool call is valid
- not valid tool call will not be present inside sample
- only sample with correct final answer will be in trainning dataset

run generation cmd:
```bash
python tests/eval/eval.py \
  --hotpotqa-path tests/data/hotpotqa/train-1000.jsonl \
  --results-path tests/eval/results_train.jsonl \
  --max-questions 0 \
  --concurrency 8
```

then extract the results to trainning ready dataset:
```bash
python tests/eval/results_extractor.py \
  --mode train \
  --results-path tests/eval/results_train.jsonl \
  --output-path tests/eval/hotpotqa_train.jsonl \
  --concurrency 20
```


## Test
- Use HotpotQA dev, and we extracted 1000 samples for test
- Our dataset and finetuned model respect the reply with "evidence" and extra "explanation", so the answer doesnt really fit the F1 score
- For the metrics, we first use another LLM to **refine** the answer given the question to align with `F1 score` so we can compare with other methods, then we apply "LLM-As-a-Judge" to judge correctness of every sample and get `Accuracy score`.

eval generation cmd:
```bash
python tests/eval/eval.py \
  --hotpotqa-path tests/data/hotpotqa/test-1000.jsonl \
  --results-path tests/eval/results_test.jsonl \
  --max-questions 0 \
  --concurrency 8
```

then extract the result and check scores:
```bash
python tests/eval/results_extractor.py \
  --mode eval \
  --results-path tests/eval/results_test.jsonl \
  --output-path tests/eval/hotpotqa_test_results.jsonl \
  --concurrency 20
```


## Guidance

Baselines:

- auto-rag: llama3-8b at `44.9`
- qwen3.5-9b with the system

Ours:

- finetuned llama3-8b with the system
- finetuned qwen3.5-9b with the system

The system uses `database_search` only at eval.
