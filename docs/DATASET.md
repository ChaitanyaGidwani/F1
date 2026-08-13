# Dataset pipeline

Five stages, each resumable, each writing an inspectable artefact.

```
collect_commons  →  presort_clip  →  make_contactsheets  →  build_dataset
   pool/            presort.jsonl      review/ sheets         dataset/
                                       ↓ human
                                    corrections.json
```

## 1. Collect - `data_pipeline.collect_commons`

```bash
python -m data_pipeline.collect_commons collect          # uses cached listing
python -m data_pipeline.collect_commons collect --refresh  # re-query Commons
python -m data_pipeline.collect_commons discover           # find more categories
```

Downloads 768px thumbnails from planned Commons categories into
`data/raw/pool/<sha1>.jpg`, with attribution in `data/raw/manifest.jsonl`.
Content-addressed filenames mean re-runs never duplicate.

**Set `COMMONS_CONTACT` to your email or project URL before running.** Wikimedia
enforces a User-Agent policy and a non-compliant agent gets throttled to
uselessness - measured, 2/12 requests succeeded versus 12/12 with a compliant
one. Details and numbers are in the module docstring.

Requests are serial with a delay. Parallel fetching gets a few percent of images
through and silently drops the rest, which is worse than slow.

### Category strategy

| Pool | Categories | Role |
|---|---|---|
| `wet` | `Formula One in rain in the <decade>s` | highest-precision wet imagery |
| `mixed` | known wet race weekends | **Damp and Drying live here** |
| `dry` | circuit categories, dry weekends | abundant, subsampled |

The `mixed` pool does the heavy lifting. A wet race weekend contains dry
practice, a wet race and the drying laps between - same circuit, same camera
positions. Framing and surface stay fixed while only the water changes, which is
the contrast the classifier needs and which a sample scattered across different
circuits cannot give.

## 2. Triage - `data_pipeline.presort_clip`

```bash
python -m data_pipeline.presort_clip
```

CLIP zero-shot sorts the pool into the four classes plus `Reject` (paddock
portraits, garages, crowds, diagrams - anything where the racing surface is not
the subject). Writes `data/raw/presort.jsonl`.

This is a **labelling aid, not the labels.** Prompts live in `app/clip_prompts.py`,
shared with the runtime fallback so the two cannot drift.

## 3. Review - `data_pipeline.make_contactsheets`

```bash
python -m data_pipeline.make_contactsheets
```

Renders 5×5 numbered grids into `data/review/`, most-confident first - those set
the class definition, so errors among them do the most damage.

Write your verdicts against the cell ids printed on each sheet, either one at a
time or a whole sheet in reading order:

```json
{
  "BN0": ["Dry", "Reject", "Damp", "Dry", "Wet"],
  "DMP012": "Wet"
}
```

Then merge them into the label store:

```bash
python -m data_pipeline.apply_review my_verdicts.json
```

That resolves cell ids to image hashes through `data/review/index_map.json` and
writes `data/review/corrections.json`, which is the durable record. Record
agreements as well as changes: "a human looked at this and agreed" is what makes
an image eligible for the test split.

### Finding Damp

Damp is not in CLIP's Damp bucket. It sits at the low-confidence boundary
between Dry and Wet in wet-weekend imagery, so there is a mode for it:

```bash
python -m data_pipeline.make_contactsheets --mode boundary --sheets 4
```

Measured yield is around 35% Damp per sheet against 23% in the bucket CLIP
itself labels "Damp". Already-decided images are skipped, so repeated runs keep
reaching new candidates.

**Why this stage is not optional.** If CLIP's guesses became ground truth, the
fine-tune would distil CLIP and the accuracy number would be circular. Human
corrections break that, and reviewed images are allocated to test/validation
first so the headline metric is measured against checked labels.

### Labelling guide

| Class | Call it this when |
|---|---|
| `Dry` | Uniform pale-grey asphalt. No sheen, no dark patches. |
| `Damp` | Uniformly darkened asphalt. No standing water, no spray, dull sheen. |
| `Drying` | Heterogeneous: a dry line through wet surface, or dry patches. Rare in practice, see below. |
| `Wet` | Standing water, mirror reflections, or visible spray off the cars. |
| `Reject` | Surface not visible or not the subject: portraits, garages, crowds, engine close-ups, diagrams, night shots too dark to judge. |

Edge cases:

- **Spray but dry-looking tarmac** → `Wet`. Spray is only produced by standing water.
- **Wet kerbs, dry racing line** → `Drying`.
- **Overcast but clearly dry surface** → `Dry`. Judge the surface, never the sky.
- **Can't tell** → `Reject`. An uncertain label is worse than a missing one.

### On `Drying`

In this corpus it barely exists. CLIP put 61 images in a "drying" bucket and none
of the 25 checked were drying tracks. Free-licence photography of a dry line
forming is rare, because photographers shoot the dramatic weather rather than the
transition.

`build_dataset.py` therefore drops any class with fewer than `--min-verified`
human-confirmed images, and Drying falls out. Keep labelling it if you see a real
one; the class reappears in the trained model once it has support. Until then the
system produces Drying from the trend layer, which is the more honest source for
it anyway.

## 4. Build - `data_pipeline.build_dataset`

```bash
python -m data_pipeline.build_dataset --max-per-class 260
```

Applies corrections over CLIP labels, drops rejects, caps per class, and writes
an `imagefolder` tree with `train`/`validation`/`test`, plus `ATTRIBUTION.csv`
and `stats.json` (counts, licence breakdown, CLIP-vs-human disagreement rate).

## 5. Demo sequence - `data_pipeline.make_demo_sequence`

```bash
python -m data_pipeline.make_demo_sequence
```

Assembles `data/demo/` - a wet → drying → damp → dry progression that the
frontend's **Load demo sequence** button plays. Real independently-labelled
Commons frames ordered into a progression, not a continuous video clip; the
trend layer consumes frame predictions and is indifferent to the difference.

## Licensing

Every image is CC-BY / CC-BY-SA / public domain from Wikimedia Commons, with
per-image credit in `ATTRIBUTION.csv`. **Keep that file with the data** -
attribution is a licence condition.

Broadcast screenshots were excluded. They would be a better domain
match and cannot be redistributed, which would make the dataset unpublishable.
