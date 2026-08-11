# Publishing to the Hugging Face Hub

Everything is scripted. You run the commands so your write token stays yours -
it is never typed into, stored by, or visible to anything in this repo.

---

## 1. Team org (satisfies the "every member has an HF account" rule)

1. Every team member creates their own account at <https://huggingface.co/join>.
2. One member creates an organisation: <https://huggingface.co/organizations/new>.
   Pick a namespace, e.g. `weather-whiplash`.
3. Invite each member: **Organization → Settings → Members → Add member**.

Hosting the dataset, model and Space under the org - rather than one personal
account - is what makes each member's individual account visible on the Hub
itself, which is what the rule asks for.

## 2. Log in locally

Activate the venv first. The `hf` CLI lives there, not on your PATH.

```bash
source .venv/bin/activate
hf auth login
```

Paste a **write** token from <https://huggingface.co/settings/tokens>. It is
stored in your own keyring by the CLI.

Verify:

```bash
hf auth whoami
```

Without activating the venv, prefix everything: `.venv/bin/hf auth login` and
`.venv/bin/python -m ...`. The older `huggingface-cli` name still works but is
deprecated.

## 3. Push the dataset

```bash
python -m data_pipeline.push_dataset --repo-id <org>/trackside-condition
```

Uploads the `imagefolder` splits, generates a dataset card from the real split
counts in `data/dataset/stats.json`, and uploads `ATTRIBUTION.csv`.

> **Keep `ATTRIBUTION.csv`.** The images are CC-BY / CC-BY-SA from Wikimedia
> Commons. Attribution is a licence condition, not a nicety.

## 4. Push the model

```bash
python -m training.push_model --repo-id <org>/vit-track-condition
```

The card is generated from `models/vit-track-condition/eval_results.json`, so
the numbers on the page are the numbers the run actually produced.

## 5. Point the app at the Hub model (optional)

```bash
WW_MODEL_ID=<org>/vit-track-condition .venv/bin/uvicorn app.main:app --port 8000
```

It is still downloaded once and held in-process - no per-request network call.

## 6. Deploy

Full walkthrough in [DEPLOY.md](DEPLOY.md). The primary target is a Docker Space
running the real backend and frontend:

```bash
python -m deploy.push_space --repo-id <org>/weather-whiplash \
                            --model-id <org>/vit-track-condition
```

Push the model first (step 4). The image bakes the weights in at build time, so
the model repo has to exist and be public before the Space builds.

A Gradio Space is available as a backup link. It shares the same trend and
recommendation code, so the two demos cannot drift apart:

```bash
python -m space.push_space --repo-id <org>/weather-whiplash-gradio \
                           --model-id <org>/vit-track-condition
```

---

## Checklist

- [ ] Org created, every member invited
- [ ] `hf auth whoami` shows the org
- [ ] Dataset page loads, card shows split counts, `ATTRIBUTION.csv` present
- [ ] Model page loads, card shows eval numbers
- [ ] `pipeline("image-classification", model="<org>/vit-track-condition")` works
      from a clean environment
- [ ] Space builds and classifies an uploaded sequence

## Troubleshooting

**403 on push** - the token is read-only, or you are not a member of the org.

**Dataset push is slow** - image bytes upload once; re-runs only send the diff.

**Space stuck building** - check the Space's build logs. The usual cause is a
`MODEL_ID` variable that points at a private model the Space cannot read; either
make the model public or add a token secret to the Space.
