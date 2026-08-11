# Deploying

The target is a **Hugging Face Docker Space** running the real FastAPI backend
and the frontend it serves, so the public link is the actual submission rather
than a cut-down demo.

Order matters: the model has to be on the Hub before the Space is built, because
the Docker build bakes the weights into an image layer.

```
push_model  ->  push_dataset  ->  deploy Space
   (Hub)          (Hub)            (pulls the model at build time)
```

---

## 0. Before you start

Activate the venv. Every command below needs it: `hf`, torch and transformers
are installed there, not system-wide.

```bash
source .venv/bin/activate
```

Log in with a **write** token from <https://huggingface.co/settings/tokens>. The
CLI stores it in your own keyring; nothing in this repo reads it.

```bash
hf auth login
hf auth whoami
```

(`huggingface-cli` still works but is deprecated in favour of `hf`. Without
activating the venv, use `.venv/bin/hf auth login`.)

Set your org once so the commands below are copy-paste:

```bash
export ORG=your-org-name
```

## 1. Push the model

```bash
python -m training.push_model --repo-id $ORG/vit-track-condition
```

The model must end up **public**, or the Space build cannot download it. (If it
has to stay private, add your token as a Space secret named `HF_TOKEN` and the
prefetch will authenticate.)

Roughly 337 MB. If it looks larger, you still have a `checkpoint-*` directory
next to the final weights; delete it first.

## 2. Push the dataset

```bash
python -m data_pipeline.push_dataset --repo-id $ORG/trackside-condition
```

## 3. Verify the container locally

Worth doing once. It catches build errors on your machine in two minutes
instead of in a Space build log in ten.

```bash
python -m deploy.push_space --stage build/space --model-id $ORG/vit-track-condition
docker build -t weather-whiplash:test build/space

docker run --rm -p 7860:7860 \
  -v "$PWD/models/vit-track-condition:/model:ro" \
  -e WW_MODEL_ID=/model \
  weather-whiplash:test
```

Open <http://127.0.0.1:7860>. The header chip should read **fine-tuned ViT**. The
bind mount is only a local shortcut so you can test before the model is on the
Hub; the deployed Space bakes the weights in at build time instead.

## 4. Deploy

```bash
python -m deploy.push_space \
  --repo-id  $ORG/weather-whiplash \
  --model-id $ORG/vit-track-condition
```

This assembles the Space tree, creates the repo, sets the `MODEL_ID` Space
variable (the Dockerfile reads it as a build arg) and uploads. The first build
takes a few minutes; watch the **Logs** tab.

## 5. Check it

- Header chip says **fine-tuned ViT**, not the CLIP fallback
- **Load demo sequence** plays and the strategy call escalates as the track dries
- `https://<space-url>/api/health` returns `"backend": "fine-tuned"`

---

## What runs where

| | Local dev | Space |
|---|---|---|
| Device | MPS | CPU |
| Model | `models/vit-track-condition` | baked into the image at build time |
| Port | 8000 | 7860 |
| Deps | `requirements.txt` | `deploy/requirements.txt`, CPU torch |

Measured 15 ms/frame on MPS and 64 ms/frame on CPU locally. A Space vCPU is
slower than an M4 core, so expect a few hundred ms there. Either is far inside
what a ~1 fps feed needs.

Note that a local `docker build` on Apple Silicon produces an arm64 image, while
Spaces build x86_64 from the same Dockerfile. The local build verifies the
Dockerfile and the dependency set, not the exact artifact HF will run.

## Troubleshooting

**Header says "CLIP zero-shot (fallback)".** The build could not fetch the
model. Check that `MODEL_ID` exists under the Space's Settings → Variables, and
that the model repo is public. The app is doing the right thing by degrading
rather than crashing, but it is not what you want in front of judges.

**Build fails on the torch install.** The Dockerfile pins CPU torch from
PyTorch's own index. Do not switch it to plain PyPI: that pulls the CUDA build,
several GB of it, onto a Space with no GPU.

**Space sleeps.** Free Spaces idle out and cold-start. Because the weights are
baked into the image there is no model download on wake, but the first request
after a sleep still pays container start. Open the link a minute before demoing.

**Port errors.** Spaces route to `app_port` in the README frontmatter, which is
7860 and must match the `CMD`.

**413 on upload.** `WW_MAX_UPLOAD_BYTES` defaults to 12 MB per image.

## Alternative: the Gradio Space

`space/push_space.py` deploys the same trend and recommendation code behind a
Gradio UI. Simpler build, plainer interface. Useful as a backup link.

```bash
python -m space.push_space --repo-id $ORG/weather-whiplash-gradio \
                           --model-id $ORG/vit-track-condition
```
