"""
Download wiki18_100w.jsonl and wiki18_100w_e5_index.zip from the
FlashRAG dataset and place them under ./tests/data.
"""

import sys

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    print("Error: modelscope library is not installed.")
    print("Install it with: pip install modelscope")
    sys.exit(1)


def main():
    dataset_id = "hhjinjiajie/FlashRAG_Dataset"
    local_dir = "./tests/data"
    file_patterns = [
        # "retrieval_corpus/wiki18_100w.jsonl",
        # "retrieval_corpus/wiki18_100w_e5_index.zip",
        # "hotpotqa/dev.jsonl",
        # "hotpotqa/train.jsonl",
	"2wikimultihopqa/dev.jsonl",
	"musique/dev.jsonl"

    ]

    try:
        snapshot_download(
            dataset_id,
            repo_type="dataset",
            allow_patterns=file_patterns,
            local_dir=local_dir,
        )
        print("Download finished. Files are in ./tests/data/retrieval_corpus/")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
