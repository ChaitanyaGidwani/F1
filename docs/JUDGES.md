# Judge brief

## What it does

A camera watches the track. Each frame is classified by surface condition. A
rolling buffer turns those frames into a direction (improving or deteriorating),
and a rule table turns the direction into a tyre call.

The work that mattered was not the model architecture. It was two things:

1. There was no dataset, so we built one, and the domain gap that made that
   necessary is measurable rather than assumed.
2. "Drying" cannot be read off a single frame. Most of the logic in this repo
   exists to pull it reliably out of a noisy classifier.

## Why we didn't just wrap an existing model

We checked before deciding to fine-tune. Searching the Hub for models:

```
BlueSR/Road_Surface_Classification     0 downloads, README only, no weights
aryaash/road-surface-classifier        7 downloads, Keras, no config.json
Gigaszi/road_surface                   1 download
```

Searching datasets across `road surface`, `wet road`, `RSCD`, `asphalt`,
`pavement`, `puddle` and `road condition` returned nothing usable for wet/dry
racing surfaces. There is no checkpoint to wrap.

## The domain gap

The closest real resource is RSCD, an autonomous-driving dataset of dry / wet /
water road patches. It is a different visual domain:

| | RSCD | Trackside / onboard |
|---|---|---|
| Camera | bumper height, car-mounted | elevated trackside, or onboard at speed |
| Framing | close-up patch, surface fills the frame | wide scene, surface is a region |
| Surface | public asphalt | racing asphalt, painted kerbs, run-off, gravel |
| Confounders | traffic, lane markings | spray plumes, marbles, a drying racing line |

The last row is the one that matters. A drying racetrack has a signature that
doesn't occur in road data: a dry racing line cut through a wet surface by the
cars themselves. It's the clearest visual cue that the tyre window is opening,
and a model trained on car-perspective road patches has never seen it.

## The dataset

873 images from Wikimedia Commons, reached through a category tree we found by
traversal rather than search: `Formula One in rain in the <decade>s`, plus known
wet race weekends (2011 Canadian GP, 2022 Monaco, 2009 Chinese, 2008 British)
and circuit categories for dry conditions.

Wet race weekends are the highest-value source. One weekend contains dry
practice, a wet race and the drying laps between, shot at the same circuit from
the same camera positions, so framing and surface stay fixed while the water
varies.

Licensing was a constraint, not an afterthought. Broadcast screenshots would be
a better domain match and were rejected because they can't be redistributed,
which would make the dataset unpublishable. Everything here is CC-BY, CC-BY-SA,
CC0 or public domain, with per-image credit in `ATTRIBUTION.csv`.

Collection had one practical lesson. Wikimedia enforces a User-Agent policy, and
a non-compliant agent gets throttled into uselessness.
Measured over 12 sequential fetches, a generic agent got 2/12 through, a
compliant one got 12/12, and a spoofed Chrome agent got 9/12. The final run
pulled 843 images with zero failures.

## What review found

CLIP zero-shot triaged the pool into candidate buckets; 322 images were then
checked by eye. **CLIP disagreed with human review on 47.8% of them.** Broken
down by bucket, the failure is very uneven:

| CLIP bucket | n | Correct | What it actually contained |
|---|---|---|---|
| Wet | 110 | 76% | mostly right; 15 were damp, not wet |
| Dry | 165 | 48% | 51 were actually damp, 32 unusable (portraits, paddock) |
| Damp | 22 | 23% | mostly Wet, some Dry, several unusable |
| **Drying** | 25 | **0%** | aerial circuit maps and sunny Monaco streets |

The Drying result is the most important finding in the project. CLIP has no
visual grounding for "a dry racing line on wet asphalt" and there is no prompt
that isolates it. Given zero verified examples, the class was dropped from the
frame classifier by an explicit rule in `build_dataset.py` (`--min-verified`)
rather than by hand, so the decision is reproducible and visible in the output.

**Drying is still produced by the system.** It comes from the trend layer, which
is where the brief itself says the real signal lives. The classifier answers
"how wet is it now"; the trend layer answers "which way is it going".

Damp needed a different approach too, and the table above shows why: only 23% of
CLIP's own "Damp" bucket was damp, while **51 damp tracks were sitting in the
"Dry" bucket**. Damp lives at the low-confidence boundary between Dry and Wet in
wet-weekend imagery, so the review tooling has a mode that samples exactly that
band (`make_contactsheets --mode boundary`). It yields around 35% Damp per
sheet, against 23% in the bucket named after it.

Final dataset: **473 images across Dry / Damp / Wet** (159 / 76 / 238), with
human-verified images allocated to test and validation first so the headline
metric is measured against checked labels.

## Why "Drying" is a sequence problem

A frame of a damp track and a frame of a drying track can be pixel-identical.
The difference is what happened in the previous ten seconds. So `app/trend.py`
does the rest, with no model involved:

- Each frame's probability vector is projected onto a wetness axis
  (`Dry 0`, `Damp 1`, `Drying 1.5`, `Wet 2`), so confidence carries through
  instead of being discarded by an argmax.
- A least-squares slope over the window says whether movement is sustained.
- The median wetness of the second half minus the first says whether it's large
  enough to matter.
- A trend is declared only when both agree in sign and clear their thresholds.

The median matters here. A single misclassified frame, from spray across the
lens or sun flaring off wet asphalt, shifts a 6-frame mean by about 0.3 wetness
units. That is enough to fake a weather change and send a car to the pits for
nothing. It shifts the median by zero. A test caught this and it's still there:
`test_single_outlier_frame_does_not_trigger_a_trend`.

This layer has no torch dependency, so it's tested against synthetic sequences
in 23 tests that run in well under a second. 66 tests across the project.

## Results

<!-- RESULTS -->
Both models on the same 69-image hand-verified test split, same label space
(`python -m training.evaluate`).

| | Accuracy | 95% CI | Macro F1 | Damp F1 | Dry F1 | Wet F1 |
|---|---|---|---|---|---|---|
| CLIP zero-shot | 75.4% | [64.0, 84.0] | 0.59 | **0.12** | 0.78 | 0.87 |
| Fine-tuned ViT | 82.6% | [72.0, 89.8] | **0.77** | **0.57** | 0.81 | 0.91 |

**The +7.2 point accuracy gap is not statistically significant, and we do not
claim it.** A McNemar paired test over the same images gives p=0.267: 9 images
the fine-tune gets right and CLIP misses, against 4 the other way. On 69 images
that is within noise, and the two confidence intervals overlap. The evaluation
script prints this verdict on every run so the number cannot quietly drift back
into a slide.

The result that does hold up is per-class:

| Class | CLIP recall | Fine-tuned recall | n | McNemar p |
|---|---|---|---|---|
| **Damp** | **0.09** | **0.55** | 11 | 0.062 |
| Dry | 0.87 | 0.83 | 23 | 1.000 |
| Wet | 0.89 | 0.91 | 35 | 1.000 |

```
CLIP zero-shot                    Fine-tuned ViT
        Damp  Dry  Wet                    Damp  Dry  Wet
Damp       1    8    2            Damp       6    3    2
Dry        2   20    1            Dry        3   19    1
Wet        2    2   31            Wet        1    2   32
```

CLIP finds **1 of 11** damp tracks. The fine-tune finds 6. CLIP has no usable
representation of the state between dry and wet, so it scatters damp tracks to
the two extremes and still scores a respectable-looking 75% overall by getting
the easy classes right. Damp is the condition where a team is on intermediates
deciding whether to stay out, so a model that cannot see it is not useful for
the decision this tool exists to support, whatever its headline accuracy.

At p=0.062 that per-class result is borderline rather than proven: a large
effect measured on 11 images. The honest reading is that the fine-tune learned
something CLIP does not have, and the test set is too small to nail it down.
More Damp labels would settle it, and that is the top item in `improvement.md`.
<!-- /RESULTS -->

The CLIP fallback stays in the code as a real, swappable path
(`WW_FORCE_CLIP=1`), not a strawman. It's what keeps the demo running if the
weights are missing.

## Limitations

- **Damp is still the thinnest class.** 71 verified examples against 102 Dry and
  96 Wet, and 11 in the test split. Photographers shoot dramatic conditions, and
  a merely damp track isn't dramatic. Training uses class weighting to
  compensate, but every Damp figure carries a wide interval: one image moves
  recall by 0.09.
- **Photographs, not video.** Commons images are composed shots. A fixed
  trackside camera has a different distribution, and the trend layer is tuned
  for roughly 1 fps.
- **Dataset skew.** Heavily Formula One and European circuits. Expect
  degradation at night, on street circuits, and in other series.
- **The recommendation table isn't race-engineer-validated.** It encodes
  plausible strategy logic, not a real team's playbook.

## Requirements

| Requirement | Where |
|---|---|
| Working frontend | `frontend/`, served by the backend at `/` |
| Working backend | `app/main.py`, model loaded in-process |
| Not a single API call | own dataset and fine-tune, no hosted inference at runtime |
| Not from-scratch | fine-tune of `google/vit-base-patch16-224-in21k` |
| Uses the HF Hub | backbone, CLIP fallback, plus our dataset, model and Space |
| Label per image | `POST /api/predict` |
| Photos **or video frames** | live camera, mp4 upload, or stills, all one path |
| Trend graph | canvas sparkline of wetness over the rolling window |
| Suggestion message | `app/recommend.py` |
| Optional weather input | `weather_hint`: raises the alert one step when a rain forecast contradicts the camera, never overrides it |

## Demo

1. `make serve PORT=8010`, open the page.
2. **Load demo sequence** for a bundled wet to dry transition. Watch the badge
   move Wet to Damp, the chart fall, and the call escalate from hold to
   "tyre change window approaching".
3. **Live camera**: point it at a screen playing race footage. A frame is
   sampled every 1.2s, so the trend moves in about fifteen seconds of video.
4. Or drop in an mp4, which is decoded and sampled across the whole clip.
5. Type "rain in 10 minutes" in the weather box on a dry track and the alert
   level rises one step. The camera still decides the condition and the tyre.
6. The header chip names which backend is live, and the chart legend marks
   Drying as coming "from trend" rather than from the classifier.
7. Restart with `WW_FORCE_CLIP=1` to run the same sequence through zero-shot.
   Watch it miss the damp frames.
