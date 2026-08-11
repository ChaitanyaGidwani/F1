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

CLIP zero-shot triaged the pool into candidate buckets; 222 images were then
checked by eye. **CLIP disagreed with human review on 48.6% of them.** Broken
down by bucket, the failure is very uneven:

| CLIP bucket | Roughly correct | What it actually contained |
|---|---|---|
| Wet | ~92% | genuinely wet track |
| Dry | ~76% | dry track, plus portraits and paddock shots |
| Damp | ~23% | mostly Wet, some Dry, several unusable |
| **Drying** | **0 of 25** | aerial circuit maps and sunny Monaco streets |

The Drying result is the most important finding in the project. CLIP has no
visual grounding for "a dry racing line on wet asphalt" and there is no prompt
that isolates it. Given zero verified examples, the class was dropped from the
frame classifier by an explicit rule in `build_dataset.py` (`--min-verified`)
rather than by hand, so the decision is reproducible and visible in the output.

**Drying is still produced by the system.** It comes from the trend layer, which
is where the brief itself says the real signal lives. The classifier answers
"how wet is it now"; the trend layer answers "which way is it going".

Damp needed a different approach too. It isn't in CLIP's Damp bucket, it's at
the low-confidence boundary between Dry and Wet in wet-weekend imagery. A
targeted pass over that band yielded Damp at roughly 35% per sheet, against 23%
in the bucket named after it.

Final dataset: **486 images across Dry / Damp / Wet**, with human-verified
images allocated to test and validation first so the headline metric is measured
against checked labels.

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
in 31 tests that run in well under a second.

## Results

<!-- RESULTS -->
Both models on the same 71-image hand-verified test split, same label space
(`python -m training.evaluate`):

| | Accuracy | Macro F1 | Damp F1 | Dry F1 | Wet F1 |
|---|---|---|---|---|---|
| CLIP zero-shot | 77.5% | 0.58 | **0.00** | 0.84 | 0.89 |
| Fine-tuned ViT | **83.1%** | **0.73** | **0.43** | 0.87 | 0.88 |

The accuracy gap is +5.6 points, but the headline number undersells it. The real
difference is Damp:

```
CLIP zero-shot                    Fine-tuned ViT
        Damp  Dry  Wet                    Damp  Dry  Wet
Damp       0    4    2            Damp       3    0    3
Dry        4   24    1            Dry        3   24    2
Wet        5    0   31            Wet        2    2   32
```

CLIP gets **0 of 6** damp tracks right. It has no usable concept of the state
between dry and wet, so it splits them between the two extremes and scores a
respectable-looking 77.5% while being useless for the decision the tool exists
to support: damp is the condition where a team is on intermediates and has to
choose whether to stay. The fine-tune gets 3 of 6, on 34 training examples.

Damp is still the weakest class, and with 6 test images the interval around 0.43
is wide. It is honest to say it went from broken to working, not from working to
good.
<!-- /RESULTS -->

The CLIP fallback stays in the code as a real, swappable path
(`WW_FORCE_CLIP=1`), not a strawman. It's what keeps the demo running if the
weights are missing.

## Limitations

- **Damp is data-starved.** 41 verified examples against 77 Dry and 64 Wet.
  Photographers shoot dramatic conditions, and a merely damp track isn't
  dramatic. Training uses class weighting to compensate, but Damp is the weakest
  class and its per-class numbers should be read with that in mind.
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
| Trend graph | canvas sparkline of wetness over the rolling window |
| Suggestion message | `app/recommend.py` |
| Optional weather input | `weather_hint`, appended, never overrides the camera |

## Demo

1. `make serve`, open the page.
2. **Load demo sequence** for a bundled wet to dry transition.
3. Watch the badge move Wet to Damp, the chart fall, and the call escalate from
   hold to "tyre change window approaching".
4. The header chip names which backend is live.
5. Restart with `WW_FORCE_CLIP=1` to run the same sequence through zero-shot.
   The difference is the argument for fine-tuning.
