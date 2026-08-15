# Sources and credits

Everything this project was built from, and where each piece came from.

---

## 1. Models

| Model | Source | Role |
|---|---|---|
| `google/vit-base-patch16-224-in21k` | Hugging Face Hub | **Backbone.** Vision Transformer pre-trained on ImageNet-21k. We fine-tuned it on our own dataset; this is the model that ships. |
| `openai/clip-vit-base-patch32` | Hugging Face Hub | **Three jobs:** triaged the raw image pool before human review, serves as the zero-shot baseline we measure against, and is the runtime fallback if the fine-tuned weights are missing. |

Our fine-tuned checkpoint: `ViTForImageClassification`, 12 layers, 768 hidden
dimensions, 3 output labels (Damp / Dry / Wet), 337 MB.

**No large language model is used anywhere.** This is an image classification
problem, not a text problem.

---

## 2. Dataset

### Where the images came from

**Wikimedia Commons**, via the MediaWiki API, across 31 categories. Nothing was
scraped from broadcast footage or a search engine.

**Wet conditions (7 categories)** — the `Formula One in rain in the <decade>s`
tree, which is the discovery that made this project possible. Commons maintains
a dedicated category for wet-weather Formula One photography, by decade:

- Formula One in rain in the 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s

**Mixed conditions (11 categories)** — known wet race weekends. These are the
highest-value source, because a single weekend contains dry practice, a wet race
and the drying laps between, shot at the same circuit from the same camera
positions. Framing and surface stay fixed while only the water changes:

- 2007 Japanese GP, 2008 British GP, 2008 Monaco GP, 2009 Chinese GP,
  2010 Chinese GP, 2010 Korean GP, 2011 Canadian GP, 2012 Malaysian GP,
  2015 United States GP, 2017 Italian GP, 2022 Monaco GP

**Dry conditions (13 categories)** — circuit and ordinary race-weekend categories:

- Silverstone Circuit, Nürburgring, Circuit de Spa-Francorchamps, Hungaroring,
  Autodromo Nazionale Monza, Circuit Paul Ricard, Red Bull Ring,
  Bahrain International Circuit, Interlagos
- 2013 Indian GP, 2015 Bahrain GP, 2019 Hungarian GP, 2019 Japanese GP

### Licences

873 images collected, 473 in the final dataset. Every one is freely
redistributable:

| Licence | Images |
|---|---|
| CC BY 2.0 | 178 |
| CC BY-SA 2.0 | 152 |
| CC BY-SA 3.0 | 50 |
| CC0 | 47 |
| CC BY-SA 4.0 | 25 |
| CC BY-SA 3.0 (nl) | 10 |
| CC BY 4.0 | 4 |
| CC BY-SA 2.0 (de), CC BY 3.0, Public domain, CC BY-SA 3.0 (de) | 7 |

Per-image title, photographer, licence and source URL are recorded in
**`data/dataset/ATTRIBUTION.csv`**, which ships with the dataset on the Hub.
Keep that file with the data — attribution is a licence condition, not a
courtesy.

### What we deliberately did *not* use

- **Broadcast screenshots (F1 TV, Sky, etc.)** — a better domain match, and
  copyrighted. Using them would have made the dataset impossible to publish,
  which breaks a hackathon deliverable.
- **RSCD (Road Surface Classification Dataset)** — referenced in our writeup as
  the closest prior work and the reason the domain gap exists, but **never
  downloaded or trained on**. It is not on the Hugging Face Hub, and it is
  car-bumper-perspective road patches rather than trackside imagery.

### Labels

Nobody else's. The images are third-party; **the labels are ours**. CLIP
zero-shot produced a first pass, then 322 images were reviewed by eye on contact
sheets and corrected. CLIP disagreed with review on 47.8% of them.

---

## 3. Software

| Package | Version | Used for |
|---|---|---|
| `torch` | 2.8.0 | Deep learning runtime |
| `torchvision` | 0.23.0 | Image augmentation during training |
| `transformers` | 4.57.6 | Model loading, `Trainer`, inference `pipeline` |
| `datasets` | 4.5.0 | `imagefolder` loading, push to Hub |
| `accelerate` | 1.10.1 | Trainer backend |
| `scikit-learn` | 1.6.1 | Accuracy, F1, confusion matrices |
| `fastapi` | 0.128.8 | Backend API |
| `uvicorn` | 0.39.0 | ASGI server |
| `av` (PyAV) | 15.1.0 | Video decoding for uploaded clips |
| `Pillow` | 11.3.0 | Image loading and contact sheets |
| `numpy` | 2.0.2 | Numerics |
| `huggingface_hub` | 0.36.2 | Hub upload and auth |
| `pytest` | 8.4.2 | 66 tests |

Frontend: **no framework, no build step, no CDN.** Plain HTML, CSS and
JavaScript; the trend chart is drawn on a `<canvas>` by hand.

Deployment: **Docker**, Python 3.11 slim base, CPU-only torch.

External services: **Wikimedia Commons API** (data collection) and the
**Hugging Face Hub** (models in, artefacts out). Nothing else is called at
runtime — inference happens in-process.

---

## 4. What we built ourselves

Everything that makes this a project rather than a wrapper:

- **The dataset** — collection pipeline, CLIP triage, contact-sheet review
  tooling, boundary-sampling mode, label store, split policy, attribution export
- **The fine-tune** — training script with class weighting for the rare Damp
  class, preprocessing aligned to the deployed inference path
- **The trend layer** (`app/trend.py`) — wetness projection, least-squares slope,
  median-based robustness, transition detection. No ML, 23 tests, and the actual
  answer to "is it drying?"
- **The recommendation engine** (`app/recommend.py`) — 12-cell rule table plus
  the forecast-conflict rule
- **The backend** (`app/main.py`) — FastAPI, session buffers, video decoding,
  upload guards
- **The frontend** — live camera capture, video upload, filmstrip, canvas chart
- **The evaluation** — CLIP baseline comparison with Wilson intervals and
  McNemar significance testing, per class

---

## 5. Our published artefacts

- Model — <https://huggingface.co/weather-whiplash/vit-track-condition>
- Dataset — <https://huggingface.co/datasets/weather-whiplash/trackside-condition>
- Code — <https://github.com/ChaitanyaGidwani/F1>

Team org on Hugging Face: `weather-whiplash` — ChaitanyaGidwani, Kritikat07,
NavyaShukla123456, C-sidd.

---

## 6. If asked "what did you actually make?"

> The backbone and the baseline are off-the-shelf from Hugging Face, and the
> photographs belong to Wikimedia Commons photographers who are credited
> individually in our attribution file. Everything else — the dataset, the
> labels, the fine-tune, the trend logic, the strategy rules, the API, the
> interface and the evaluation — is ours.
