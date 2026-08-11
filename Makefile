PY := .venv/bin/python
PORT ?= 8000
ORG ?= your-org

.PHONY: help setup test serve collect triage sheets dataset demo train eval \
        push-dataset push-model push-space clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## create the venv and install dependencies
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:  ## run the test suite
	$(PY) -m pytest tests/ -q

serve:  ## run the app (frontend + API) on $(PORT)
	.venv/bin/uvicorn app.main:app --port $(PORT) --host 127.0.0.1

# --- data pipeline (see docs/DATASET.md) ---

collect:  ## download the Commons image pool
	COMMONS_CONTACT=$${COMMONS_CONTACT:-set-your-email} $(PY) -m data_pipeline.collect_commons collect

triage:  ## CLIP zero-shot pre-sort into candidate buckets
	$(PY) -m data_pipeline.presort_clip

sheets:  ## render contact sheets for human review
	$(PY) -m data_pipeline.make_contactsheets

dataset:  ## apply corrections and build the imagefolder splits
	$(PY) -m data_pipeline.build_dataset

demo:  ## assemble the bundled wet -> dry demo sequence
	$(PY) -m data_pipeline.make_demo_sequence

# --- model ---

train:  ## fine-tune the ViT backbone
	$(PY) -m training.train_vit --epochs 8

eval:  ## fine-tuned vs CLIP zero-shot on the held-out test split
	$(PY) -m training.evaluate

# --- hub (see docs/PUSH_TO_HUB.md; run `huggingface-cli login` first) ---

push-dataset:  ## push the dataset  (make push-dataset ORG=my-org)
	$(PY) -m data_pipeline.push_dataset --repo-id $(ORG)/trackside-condition

push-model:  ## push the fine-tuned model
	$(PY) -m training.push_model --repo-id $(ORG)/vit-track-condition

push-space:  ## deploy the Gradio Space
	$(PY) -m space.push_space --repo-id $(ORG)/weather-whiplash \
		--model-id $(ORG)/vit-track-condition

clean:  ## remove build artefacts (keeps the downloaded pool)
	rm -rf .pytest_cache **/__pycache__ data/dataset
