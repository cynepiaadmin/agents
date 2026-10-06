# ─────────────────────────────────────────────────────────────────────────
# browser-agent — build / test the A2E browser agent.
#
# The build context is the PARENT directory (the one holding browser-agent/,
# browser-env/ and a2eprotocol/) so the a2e SDK checkout is in the image. Both
# browser repos MUST share one context — they pin the SDK from the same local
# checkout, and building separately would put the SDK out of reach.
# ─────────────────────────────────────────────────────────────────────────

SHELL       := /bin/bash
IMAGE_NAME  := browser-agent
VERSION_TAG := latest
REGISTRY    ?=
PORT        ?= 8770

REPO_ROOT   := ..
DOCKERFILE  := Dockerfile

.PHONY: all build run stop clean rebuild config test test-local venv help

all: build

build: ## Build the image (context = repo root)
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME):$(VERSION_TAG) $(REPO_ROOT)

tag: ## Tag for a registry push (make tag REGISTRY=registry.example.com)
ifdef REGISTRY
	docker tag $(IMAGE_NAME):$(VERSION_TAG) $(REGISTRY)/$(IMAGE_NAME):$(VERSION_TAG)
else
	@echo "Usage: make tag REGISTRY=registry.example.com"
endif

push: tag ## Push the tagged image
ifdef REGISTRY
	docker push $(REGISTRY)/$(IMAGE_NAME):$(VERSION_TAG)
else
	@echo "Usage: make push REGISTRY=registry.example.com"
endif

run: ## Run an episode with the fixed policy against a host at http://localhost:$(PORT)
	XA_ENV_BASE_URL=http://localhost:$(PORT) \
	  python -m browser_agent.cli run --task add-widget-to-cart

stop: ## Stop a running container
	-docker stop $(IMAGE_NAME) 2>/dev/null; docker rm $(IMAGE_NAME) 2>/dev/null; true

clean: ## Remove the image
	-docker rmi $(IMAGE_NAME):$(VERSION_TAG) 2>/dev/null; true

rebuild: clean build

venv: ## Create a local venv with the SDK + this package installed (dev)
	uv venv .venv --python 3.11
	uv pip install --python .venv/bin/python -e ../a2eprotocol/python-sdk
	uv pip install --python .venv/bin/python -e ".[dev]"

test: ## Run the test suite (spawns browser-env as a subprocess; no LLM needed)
	pytest -q

test-local: ## Same, via the local venv
	.venv/bin/pytest -q

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'