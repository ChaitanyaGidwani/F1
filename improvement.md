# Review against the hackathon brief

Audit date: 11 Aug 2026. Every number below was re-measured from the repo, not
copied from earlier notes.

---

## Status: P0 and P1 addressed

Everything below the line was the original audit. This is what changed after it.

| # | Item | Status |
|---|---|---|
| P0-1 | Significance overclaim | **Fixed.** Judge brief rewritten; McNemar and Wilson intervals now print on every `training.evaluate` run |
| P0-2 | Damp under-supported | **Improved.** 46 -> 76 images, test split 6 -> 11. Recall 0.09 -> 0.55 against CLIP |
| P0-3 | No video input | **Fixed.** `app/video.py`, decoded and sampled across the whole clip, 10 tests |
| P1-4 | No live camera | **Added.** Frame every 1.2s from `getUserMedia` into the existing session buffer |
| P1-5 | No hosted link | **Won't do.** HF Spaces now charges for any backend; deployment stays built and verified in `deploy/` |
| P1-6 | Four-vs-three classes buried | **Fixed.** Top of README, and the UI legend marks Drying "from trend" |
| P1-7 | Weather input decorative | **Fixed.** A rain note that contradicts the camera raises the alert one step, never the tyre |
| P2 | Frame cap, pixel bomb, CORS | **Fixed.** 60-frame cap, `DecompressionBombError` caught, CORS opt-in |

Numbers after the work: **473 images** (Dry 159 / Damp 76 / Wet 238), 322 human
decisions, **66 tests**. On the 69-image hand-verified test split the fine-tune
scores 82.6% against CLIP's 75.4%, which is *not* significant (McNemar p=0.267)
and is no longer claimed. Damp recall 0.09 -> 0.55 (p=0.062) is the result that
carries the argument.

### Still open

- **Damp remains the thinnest class.** 11 test images means one image moves
  recall by 0.09, and the p=0.062 result sits just outside significance. Another
  two boundary review passes would likely settle it; 124 candidates remain and
  `make_contactsheets --mode boundary` finds them.
- **The Hub copies are stale.** The model and dataset were retrained and rebuilt
  after they were pushed. Re-push before judging (see below).
- **Drying still has no frame-level training data.** Correct call on the
  evidence, but if a source of dry-line imagery turns up it is worth revisiting.

---

## Original audit

## Verdict

**Competitive, not yet a lock.**

The engineering rigour is the strongest thing here and it is genuinely unusual
for a hackathon: a dataset built from nothing with a defensible sourcing story,
a measured reason behind every design decision, negative results reported rather
than buried, and 44 tests including a trend layer that is testable without a
model. Against a field of API wrappers this stands out.

What stops it being a lock is that the parts a judge *sees in five minutes* have
holes. There is no video input, no live camera, no clickable link, and the model
covers three of the four classes named in the problem statement. A flashier
submission with a webcam feed and a public URL can beat this on presentation
while being technically thinner.

One item below is not a gap but a **correctness problem in our own claims**, and
it should be fixed before anyone presents this.

---

## Compliance against the brief

| Requirement | Status | Evidence |
|---|---|---|
| Working frontend | Pass | `frontend/`, served at `/`, verified in browser |
| Working backend | Pass | `app/main.py`, model in-process, 13 API tests |
| Not a single API call | Pass | own dataset + fine-tune, no hosted inference at runtime |
| Not from-scratch | Pass | fine-tune of `google/vit-base-patch16-224-in21k` |
| Uses `transformers.Trainer` | Pass | `training/train_vit.py` |
| Uses the HF Hub | Pass | backbone, CLIP fallback, plus our model + dataset |
| Dataset pushed to Hub, org namespace | Pass | `weather-whiplash/trackside-condition` |
| Model pushed to Hub, org namespace | Pass | `weather-whiplash/vit-track-condition` |
| Every member has own HF account | Pass | org has 4 members: ChaitanyaGidwani, Kritikat07, NavyaShukla123456, C-sidd |
| CLIP zero-shot fallback, swappable | Pass | `WW_FORCE_CLIP=1` |
| Trend logic as testable function | Pass | `app/trend.py`, 31 tests, no model needed |
| Rule-based recommendation | Pass | `app/recommend.py`, 12-cell table |
| Trend works across a sequence | Pass | demo escalates HOLD to PREPARE to ACT |
| Judge-facing domain-gap writeup | Pass | `docs/JUDGES.md` |
| **~100-200 images per class** | **Partial** | Dry 199, Wet 241, **Damp 46** |
| **Four classes: Dry/Damp/Wet/Drying** | **Partial** | classifier has 3; Drying comes from the trend layer |
| **Input: photos or video frames** | **Fail** | `accept="image/*"`; no video decoding anywhere |
| Optional: HF Space | Not done | blocked by HF paywall, deployment built and verified |
| Optional: weather info input | Cosmetic | free-text only, never parsed or used |

---

## P0 — fix before presenting

### 1. The headline accuracy claim is not statistically significant

`docs/JUDGES.md` presents "+5.6 points" as the result of fine-tuning. Re-tested
with a paired McNemar test on the same 71 test images:

```
ViT correct & CLIP wrong : 10
CLIP correct & ViT wrong : 6
exact two-sided p        : 0.454     -> not significant at 0.05
```

Wilson intervals confirm it: CLIP 77.5% [66.5, 85.6], fine-tuned 83.1%
[72.7, 90.1]. The intervals overlap almost entirely. **On overall accuracy we
cannot claim the fine-tune is better.** A judge with a statistics background
will find this in one question, and being caught overclaiming costs more than
the claim was ever worth.

The honest and still-strong framing is the capability gap, which does not depend
on the accuracy delta:

> CLIP gets **0 of 6** damp tracks right, the fine-tune gets 3. CLIP has no
> usable representation of the state between dry and wet, so it splits damp
> tracks between the two extremes. Damp is exactly the condition where a team is
> on intermediates deciding whether to stay out. Overall accuracy is
> statistically indistinguishable on a test set this small; the difference that
> matters is that one model can represent the middle class and the other cannot.

**Action:** rewrite the Results section of `docs/JUDGES.md` to lead with the
Damp capability gap, quote accuracy with confidence intervals, and state the
McNemar result plainly. Add the paired test to `training/evaluate.py` so it is
reported every run rather than being a one-off.

**Effort:** 30 minutes. **Impact:** removes the only place the submission is
vulnerable to being called dishonest.

### 2. Damp is under-supported

46 images total against the brief's 100-200 per class, and only 6 in the test
split. Every Damp number carries an enormous error bar, and it is the class the
product exists to detect.

**Action:** run two or three more boundary review passes. The targeted method
already works — sampling the low-confidence Dry/Wet band from wet-weekend
imagery yielded Damp at ~35% per sheet against 23% in CLIP's own Damp bucket.
174 unreviewed boundary candidates remain, so roughly 60 more Damp images are
reachable in about an hour of review, then retrain.

**Effort:** 1-2 hours. **Impact:** turns the weakest class into a defensible one
and lets you raise the test split above the point where one image moves F1 by
0.08.

### 3. No video input

The brief says "Photos or **video frames** of the track" and "images or short
video frames from a camera". A judge who drags an `.mp4` onto the page gets
nothing. Right now `frontend/index.html` sets `accept="image/*"` and no code
path decodes video.

**Action:** accept video in the file input, decode server-side, sample every Nth
frame, and feed the existing `/api/sequence` path. `av` or `opencv-python-headless`
both work; `av` is the lighter dependency. Roughly:

```python
# app/video.py
import av
def frames_from_video(raw: bytes, every_n: int = 15, cap: int = 40):
    container = av.open(io.BytesIO(raw))
    for i, frame in enumerate(container.decode(video=0)):
        if i % every_n == 0:
            yield frame.to_image()
```

**Effort:** 1-2 hours including a test. **Impact:** closes the one outright
requirement failure, and a video playing through the trend chart is a much
better demo than clicking through stills.

---

## P1 — high value for a five-minute pitch

### 4. No live camera mode

The product is called a *live* track condition detector and is demonstrated with
a folder of JPEGs. A webcam feed sampling one frame per second into
`/api/predict` would need no backend changes at all — the session buffer already
does exactly this.

```js
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
// draw to canvas every 1000ms, canvas.toBlob(), POST to /api/predict
```

Point a laptop at a phone screen playing race footage and the trend line moves
live. That is the demo that gets remembered.

**Effort:** 1-2 hours, frontend only. **Impact:** highest presentation return of
anything on this list.

### 5. No clickable link

HF Spaces now charges for any Space with a backend. The Docker deployment in
`deploy/` is built and verified locally, so this is purely a hosting decision:
HF PRO at $9/month is one command away, or Google Cloud Run's free tier fits the
image. Judges reward a link they can open on their own phone.

**Effort:** 10 minutes with PRO, 30-45 minutes with Cloud Run.

### 6. "Drying" is missing from the classifier and the framing is buried

Dropping it was correct — 0 of 25 CLIP-assigned images were drying tracks — and
the reasoning is documented. But a judge reading the problem statement sees four
class names and our model outputs three. This has to be the *first* thing said
about classes, not a note in a limitations section, or it reads as incomplete
rather than deliberate.

Two options, in order of value:

- **Reframe (30 min, safe):** put the four-vs-three explanation at the top of
  the README and the judge brief. "The system reports four states. Three come
  from the classifier, Drying comes from the trend layer, because a damp track
  and a drying track are pixel-identical in one frame."
- **Recover the class (3+ hours, risky):** hand-hunt dry-line imagery. Free
  licence photography of a forming dry line is genuinely rare. Not worth it
  before the deadline.

### 7. Weather input is decorative

`weather_hint` is appended to the message string and never parsed. The brief
lists weather as optional, so this is not a failure, but a judge may well ask
"what does the weather box actually do?" and the honest answer today is
"nothing". Either wire it to a real forecast lookup that raises urgency when
rain is imminent, or relabel it as an operator note so no one expects more.

**Effort:** 20 minutes to relabel, 2 hours to make it real.

---

## P2 — robustness and security

None of these will lose the hackathon. They matter if this is shown as
production-shaped work.

| Finding | Where | Risk | Fix |
|---|---|---|---|
| No cap on frames per request | `app/main.py` `/api/sequence` | Uploading 500 frames blocks a worker for ~30s. Trivial DoS, and an easy accident during a demo. | Reject above ~60 files with a 400 |
| `DecompressionBombError` uncaught | `app/main.py` `read_image` | A pixel-bomb image returns 500 instead of a clean 400; `except (UnidentifiedImageError, OSError)` does not catch it | Add `Image.DecompressionBombError` to the except clause |
| CORS is `allow_origins=["*"]` | `app/main.py` | Any site can call the API. Harmless on localhost, wrong if ever hosted | Restrict to the deploy origin when hosting |
| No auth or rate limiting | all endpoints | Fine for a demo, unacceptable if the Space ever goes public | Note as out of scope, or add a simple token |
| Test split is 71 images | `data/dataset` | Every metric has a ~17pp confidence interval | Grows naturally from item 2 |

---

## Suggested order of work

If there is one day left:

1. Fix the significance overclaim (30 min) — **do this first, it is a defect**
2. Live webcam mode (1-2 h) — biggest demo win
3. Video upload (1-2 h) — closes the requirement failure
4. More Damp labels and retrain (1-2 h)
5. Reframe the four-vs-three class story up front (30 min)
6. P2 robustness fixes (30 min total)

If there are two hours left: items 1, 5, and the two-line P2 fixes. They are
cheap and they remove every place where the submission can be argued with.

---

## What to say to judges

Lead with the thing nobody else will have: **you built the dataset because the
Hub had nothing, and you can prove it.**

- No wet/dry racetrack model or dataset exists on the Hub. Searches across
  `road surface`, `RSCD`, `asphalt`, `pavement`, `puddle` return nothing usable.
- 873 CC-licensed images from Wikimedia Commons, found through a category tree
  (`Formula One in rain in the <decade>s`) rather than search. Broadcast frames
  were rejected on purpose because they cannot be redistributed.
- CLIP triage disagreed with human review on **48.6% of 222 checked images**.
  Its entire "Drying" bucket was aerial circuit maps and sunny streets.
- So the classifier answers "how wet is it now", and a separate model-free trend
  layer answers "which way is it going" — because a damp frame and a drying
  frame are identical, and only the sequence separates them.
- The trend layer uses a median rather than a mean because one misclassified
  frame shifts a 6-frame mean by ~0.3 wetness units, enough to fake a weather
  change and pit a car for nothing. A test caught that.

Then be first to name the limits: Damp is data-starved, the test set is small
enough that overall accuracy differences are not significant, and Drying is a
trend state rather than a frame label. Judges trust a team that volunteers its
error bars far more than one that has to be asked for them.
