# Weather Whiplash

Track conditions change faster than weather reports. This reads frames from a
trackside or onboard camera, classifies the racing surface, tracks whether
things are getting better or worse, and turns that into a tyre call.

Built for the Grand Prix hackathon (AI in Racing Strategy & Decision-Making,
problem statement 2).

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000
```

Open <http://127.0.0.1:8000>. Drop in some track frames: one image classifies a
single frame, several images replay as a sequence so the trend can move. There's
a **Load demo sequence** button with a bundled wet-to-dry transition.

The backend serves the frontend, so that's the whole setup. No build step, no
CDN, one port. If the fine-tuned weights are missing it falls back to CLIP
zero-shot and says so in the header instead of failing.

`make help` lists the pipeline steps.

## How it works

```
frames -> ViT classifier -> rolling buffer -> trend analysis -> rule table -> call
          (fine-tuned)      (last N frames)   slope + median    (if/else)
```

**Frame classification.** `google/vit-base-patch16-224-in21k` fine-tuned on
trackside imagery, loaded in-process at startup. No hosted-inference call, so
serving a request needs no network.

**Trend detection** (`app/trend.py`) is the part that answers the actual
question. A single frame can't express "drying". Each frame's probability vector
is projected onto a wetness axis (`Dry 0`, `Damp 1`, `Drying 1.5`, `Wet 2`), and
the buffer is reduced to a direction using two signals that have to agree:

- a least-squares **slope** over the window, for whether movement is sustained
- the **median** wetness of the second half minus the first, for whether it's
  big enough to matter

Requiring both is what stops a classifier flickering between adjacent classes
from reading as a weather event. One misclassified frame moves a 6-frame mean by
about 0.3 wetness units, enough to fake a trend; it moves the median by nothing.
There's a test for exactly that case.

**Recommendation** (`app/recommend.py`) is a lookup table from
`(trend, condition)` to a message, an action and a tyre. It's twelve cells of
domain knowledge an engineer can read and argue with.

## Classes

The system reports **Dry, Damp, Drying, Wet**. The first three come from
different places, and it's worth being precise about which:

- The classifier is trained on **Dry / Damp / Wet**, the states observable in a
  single frame.
- **Drying** comes from the trend layer, not the classifier. When we reviewed
  the data, CLIP's entire "Drying" bucket turned out to be aerial circuit maps
  and sunny Monaco streets, with zero real drying tracks among 25 checked. With
  no verified examples, training the class would have meant training on noise,
  so `build_dataset.py` drops any class without human-verified support.

That split matches how the signal actually behaves. A drying track and a damp
track can look identical in one frame; the difference is the direction of travel
across a sequence.

## The dataset

No wet/dry racetrack classifier or dataset exists on the Hub, which is why this
needs a fine-tune rather than an API call. Full reasoning in
[docs/JUDGES.md](docs/JUDGES.md); the short version:

- RSCD, the nearest resource, is car-mounted close-up road patches. A trackside
  camera is wide, elevated, and full of confounders RSCD never sees.
- So the dataset is built from Wikimedia Commons motorsport photography, mainly
  the `Formula One in rain in the <decade>s` category tree plus known wet race
  weekends. CC-licensed, redistributable, fully attributed.
- Broadcast screenshots were ruled out: they cannot be published to
  the Hub, which would break the deliverable.

Labels come from CLIP triage corrected by human review. CLIP disagreed with the
reviewer on 48.6% of 222 checked images. Verified images go to the test and
validation splits first, so accuracy is measured against checked labels.

Rebuilding it:

```bash
python -m data_pipeline.collect_commons collect   # download pool
python -m data_pipeline.presort_clip              # CLIP triage
python -m data_pipeline.make_contactsheets        # review grids
#   ... check the sheets, write data/review/corrections.json ...
python -m data_pipeline.build_dataset             # imagefolder + splits
python -m training.train_vit --epochs 10
python -m training.evaluate                       # vs CLIP zero-shot
```

See [docs/DATASET.md](docs/DATASET.md) for the labelling guide.

## Publishing to the Hub

Dataset, model and Space push under your org namespace. Walkthrough including
org setup: [docs/PUSH_TO_HUB.md](docs/PUSH_TO_HUB.md).

```bash
huggingface-cli login
python -m data_pipeline.push_dataset --repo-id <org>/trackside-condition
python -m training.push_model       --repo-id <org>/vit-track-condition
python -m space.push_space          --repo-id <org>/weather-whiplash \
                                    --model-id <org>/vit-track-condition
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | backend, device, model id, class list |
| `POST` | `/api/predict` | one frame, folded into the session trend |
| `POST` | `/api/sequence` | N ordered frames, per-frame trace plus verdict |
| `POST` | `/api/session/reset` | clear a session's buffer |
| `GET`  | `/api/demo-sequence` | bundled wet to dry frames |

```bash
curl -F "file=@frame.jpg" -F "session_id=pitwall" http://127.0.0.1:8000/api/predict
```

Sessions are per-camera rolling buffers, LRU-capped so a long demo can't leak
memory.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Covers the trend logic against synthetic sequences (no model needed), the rule
table, and the API against a scripted classifier, so a specific weather story is
asserted to produce a specific strategy call.

## Layout

```
app/              backend: trend logic, rules, inference, FastAPI
  trend.py          rolling buffer -> DRYING / WETTING / STABLE
  recommend.py      (trend, condition) -> tyre call
  inference.py      fine-tuned ViT, with CLIP zero-shot fallback
  main.py           API + static hosting
frontend/         vanilla JS UI, canvas trend chart
data_pipeline/    collect -> triage -> review -> build -> push
training/         fine-tune, evaluate, push
space/            Gradio Space, shares app/ so the logic exists once
tests/            trend, rules, API
docs/             judge brief, dataset notes, Hub walkthrough
```

## Configuration

Everything is an environment variable, see `app/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `WW_MODEL_ID` | `models/vit-track-condition` | local path or Hub id |
| `WW_FORCE_CLIP` | `0` | force the zero-shot fallback |
| `WW_DEVICE` | `auto` | mps, then cuda, then cpu |
| `WW_WINDOW` | `12` | rolling buffer length |
| `WW_SLOPE_THRESHOLD` | `0.025` | trend sensitivity |
