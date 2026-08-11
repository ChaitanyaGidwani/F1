PY := .venv/bin/python
PORT ?= 8000
ORG ?= your-org

SPACE ?= weather-whiplash
MODEL ?= vit-track-condition

.PHONY: help setup test serve collect triage sheets dataset demo train eval \
        login whoami push-dataset push-model push-space stage-space docker-test deploy clean

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

# --- hub (see docs/PUSH_TO_HUB.md; run `make login` first) ---

login:  ## log in to Hugging Face (uses the venv's hf CLI)
	.venv/bin/hf auth login

whoami:  ## show which Hugging Face account is logged in
	.venv/bin/hf auth whoami

push-dataset:  ## push the dataset  (make push-dataset ORG=my-org)
	$(PY) -m data_pipeline.push_dataset --repo-id $(ORG)/trackside-condition

push-model:  ## push the fine-tuned model
	$(PY) -m training.push_model --repo-id $(ORG)/vit-track-condition

push-space:  ## deploy the Gradio Space (backup demo link)
	$(PY) -m space.push_space --repo-id $(ORG)/$(SPACE)-gradio \
		--model-id $(ORG)/$(MODEL)

# --- deploy the real app as a Docker Space (see docs/DEPLOY.md) ---

stage-space:  ## assemble the Space tree into build/space
	$(PY) -m deploy.push_space --stage build/space --model-id $(ORG)/$(MODEL)

docker-test: stage-space  ## build and run the Space image locally on :7860
	docker build -t $(SPACE):test build/space
	docker run --rm -p 7860:7860 \
		-v "$(PWD)/models/$(MODEL):/model:ro" -e WW_MODEL_ID=/model $(SPACE):test

deploy:  ## deploy the Docker Space  (make deploy ORG=my-org)
	$(PY) -m deploy.push_space --repo-id $(ORG)/$(SPACE) --model-id $(ORG)/$(MODEL)

clean:  ## remove build artefacts (keeps the downloaded pool)
	rm -rf .pytest_cache **/__pycache__ data/dataset build/
