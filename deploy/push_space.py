"""Deploy the FastAPI app and its frontend as a Hugging Face Docker Space.

Judges get the real submission at a public URL, not a reduced stand-in: the same
backend, the same UI, the same trend code.

The Space tree is assembled into a staging directory first, so the thing tested
locally is byte-for-byte the thing uploaded:

    python -m deploy.push_space --stage build/space     # assemble only
    docker build -t ww build/space                      # verify it builds
    docker run -p 7860:7860 ww                          # verify it runs

Then deploy (run `hf auth login` first so your token stays yours):

    python -m deploy.push_space --repo-id <org>/weather-whiplash \\
                                --model-id <org>/vit-track-condition
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"

README = """---
title: Weather Whiplash
emoji: 🏁
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Weather Whiplash - live track condition detector

Reads frames from a trackside or onboard camera, classifies the racing surface
as **Dry / Damp / Wet**, tracks whether conditions are improving or
deteriorating, and turns that into a tyre call.

Press **Load demo sequence** for a bundled wet-to-dry transition, or drop in
your own frames. A single image classifies one frame; several images replay as a
sequence, which is what lets the trend layer move.

**Drying is a trend state, not a frame label.** A damp track and a drying track
can look identical in one image; the difference is the direction of travel
across a sequence. The classifier answers "how wet is it now" and a separate
rolling-window layer answers "which way is it going".

{model_line}
"""


def stage(dest: Path, model_id: Optional[str]) -> Path:
    """Assemble the exact tree that the Space will contain."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    shutil.copytree(ROOT / "app", dest / "app", ignore=ignore)
    shutil.copytree(ROOT / "frontend", dest / "frontend", ignore=ignore)
    shutil.copytree(ROOT / "data" / "demo", dest / "data" / "demo", ignore=ignore)

    shutil.copyfile(DEPLOY / "Dockerfile", dest / "Dockerfile")
    shutil.copyfile(DEPLOY / "requirements.txt", dest / "requirements.txt")

    model_line = (
        f"Model: [`{model_id}`](https://huggingface.co/{model_id})"
        if model_id
        else "_No fine-tuned model configured; running the CLIP zero-shot fallback._"
    )
    (dest / "README.md").write_text(README.format(model_line=model_line))

    files = sum(1 for _ in dest.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"staged {files} files ({size / 1e6:.1f} MB) -> {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="e.g. my-org/weather-whiplash")
    parser.add_argument("--model-id", help="Hub id of the fine-tuned model")
    parser.add_argument("--stage", metavar="DIR",
                        help="assemble the Space tree here and exit")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if args.stage:
        stage(Path(args.stage), args.model_id)
        return

    if not args.repo_id:
        raise SystemExit("--repo-id is required (or use --stage to assemble only)")
    if not args.model_id:
        print("[!] no --model-id: the Space will run the CLIP zero-shot fallback")

    from huggingface_hub import HfApi

    staged = stage(ROOT / "build" / "space", args.model_id)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="space", space_sdk="docker",
                    exist_ok=True, private=args.private)

    # The variable has to exist before the upload, because uploading is what
    # triggers the build and the Dockerfile reads MODEL_ID as a build arg.
    if args.model_id:
        api.add_space_variable(args.repo_id, "MODEL_ID", args.model_id)

    api.upload_folder(
        folder_path=str(staged),
        repo_id=args.repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "*.pyc"],
    )

    print(f"\ndeployed -> https://huggingface.co/spaces/{args.repo_id}")
    print("The first build takes a few minutes; watch the Logs tab.")


if __name__ == "__main__":
    main()
