"""Deploy the Gradio demo as a Hugging Face Space.

Uploads `space/app.py` as the Space entrypoint plus the shared `app/` package,
so the Space runs the exact same trend and recommendation code as the backend.

    huggingface-cli login
    python -m space.push_space --repo-id <your-org>/weather-whiplash \\
        --model-id <your-org>/vit-track-condition
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

README = """---
title: Weather Whiplash
emoji: 🏁
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Weather Whiplash - Live Track Condition Detector

Classifies trackside racing-surface frames as **Dry / Damp / Drying / Wet**,
then reads a rolling window of frames to decide whether the track is improving
or deteriorating and when the tyre-change window opens.

Upload a numbered sequence of frames to see the trend logic work - a single
frame cannot express "drying".

Model: [`{model_id}`](https://huggingface.co/{model_id})
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-id", required=True, help="Hub id of the fine-tuned model")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(
        args.repo_id, repo_type="space", space_sdk="gradio",
        exist_ok=True, private=args.private,
    )

    api.upload_file(
        path_or_fileobj=str(ROOT / "space" / "app.py"),
        path_in_repo="app.py",
        repo_id=args.repo_id,
        repo_type="space",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "space" / "requirements.txt"),
        path_in_repo="requirements.txt",
        repo_id=args.repo_id,
        repo_type="space",
    )
    api.upload_file(
        path_or_fileobj=README.format(model_id=args.model_id).encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="space",
    )
    # The shared logic package - same code as the FastAPI backend.
    api.upload_folder(
        folder_path=str(ROOT / "app"),
        path_in_repo="app",
        repo_id=args.repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "main.py"],
    )

    api.add_space_variable(args.repo_id, "MODEL_ID", args.model_id)

    print(f"done -> https://huggingface.co/spaces/{args.repo_id}")
    print("Set MODEL_ID in Space settings if the variable did not apply.")


if __name__ == "__main__":
    main()
