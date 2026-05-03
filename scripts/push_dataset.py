"""Push the LawMate QA + KB dataset to HuggingFace Hub.

Repo: https://huggingface.co/datasets/Tamir39/vietnam-tax-qa (public, CC-BY-4.0).

Run once after `huggingface-cli login` (or with HF_TOKEN env set):

    python scripts/push_dataset.py
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "Tamir39/vietnam-tax-qa"
REPO_TYPE = "dataset"

UPLOADS = [
    # (local path, path inside the repo)
    (ROOT / "data" / "knowledge_base" / "knowledge_base.csv", "knowledge_base/knowledge_base.csv"),
    (ROOT / "data" / "qa" / "train_qa.jsonl",                 "qa/train_qa.jsonl"),
    (ROOT / "data" / "qa" / "test_qa.jsonl",                  "qa/test_qa.jsonl"),
    (ROOT / "data" / "HF_README.md",                          "README.md"),
]


def main() -> None:
    api = HfApi()
    print(f"whoami: {api.whoami()['name']}")

    for src, _ in UPLOADS:
        if not src.is_file():
            raise FileNotFoundError(src)

    create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)
    print(f"repo ready: https://huggingface.co/datasets/{REPO_ID}")

    for src, dest in UPLOADS:
        print(f"  uploading {src.name} -> {dest}")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"upload {dest}",
        )

    print(f"\ndone -> https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
