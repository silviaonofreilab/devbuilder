# DevBuilder operator entry point.
#
# Usage: make <target> [CONFIG=configs/<file>.yaml]
#
# Variables:
#   CONFIG   YAML config file (default: configs/settings.yaml)
#   TARGET   Which API the api-* smoke targets talk to: local | vps
#            (default: local). `vps` uses VPS_API_TOKEN and assumes
#            `make droplet-tunnel` is forwarding localhost:8080.

CONFIG  ?= configs/settings.yaml
TARGET  ?= local
ifeq ($(TARGET),vps)
TOKEN_VAR = VPS_API_TOKEN
else
TOKEN_VAR = API_TOKEN
endif
TAG ?= 0.2.3
CLOUDFLARE_TAG ?= 2026.5.0
VPS_TAGS = API_TAG=$(TAG) EXPORT_TAG=$(TAG) INDEXER_TAG=$(TAG) UI_TAG=$(TAG) CLOUDFLARE_TAG=$(CLOUDFLARE_TAG)
LOAD_ENV = set -a; . ./.env; set +a;

# Provisioning goes through one CLI that loads .env itself and records what
# it created in .devbuilder/state.json. Make never writes state; it reads
# single fields with `get` when a recipe needs an endpoint or a host.
DEPLOY = uv run devbuilder-deploy --config $(CONFIG)
STATE_GET = uv run devbuilder-deploy get
# Keep interactive sessions and the port-forward alive through idle periods.
SSH_OPTS = -o ServerAliveInterval=60 -o ServerAliveCountMax=3
require_droplet = @test -n "$(DROPLET_IP)" || { echo "No droplet in state; run 'make droplet-up' first." >&2; exit 1; }
# The vps-* targets only make sense from the synced copy on the droplet.
require_vps = @test "$$(pwd)" = "/opt/devbuilder" || { echo "vps-* targets run on the droplet: make droplet-ssh, then cd /opt/devbuilder." >&2; exit 1; }

export KMP_DUPLICATE_LIB_OK = TRUE

# Strict recipes: fail on the first error, on unset variables, and on any
# failing stage of a pipeline (so `curl ... | json.tool` cannot mask a 503).
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

COMPOSE = docker compose
COMPOSE_LOCAL = $(COMPOSE) --profile local
COMPOSE_VPS = $(COMPOSE) -f compose.yaml -f compose.vps.yaml --profile vps

.DEFAULT_GOAL := help

.PHONY: help \
		setup lint format \
		parse chunk pipeline indexer-local query query-rag \
		export-encoder export-reranker \
		qdrant-up qdrant-down \
		api-build api-up api-down api-logs api-shell api-ready api-ask-smoke api-retrieve-smoke \
		indexer-run export-run \
		stack-up stack-down stack-logs stack-smoke \
		status pod-up pod-status pod-stop pod-terminate \
		droplet-up droplet-status droplet-sync-config droplet-ssh droplet-tunnel droplet-destroy \
		remote-ghcr-login remote-ghcr-logout vps-export vps-indexer vps-up vps-down \
		droplet-sync-data \
		ui-build ui-up ui-down ui-logs ui-vps \
		test \
		clean clean-all teardown

#### Help ####

help:
	@printf "DevBuilder targets:\n"
	@printf "  %-20s %s\n" "Setup:"              "setup, lint, format"
	@printf "  %-20s %s\n" "Data:"               "parse, chunk, pipeline"
	@printf "  %-20s %s\n" "Models (bare):"      "export-encoder, export-reranker"
	@printf "  %-20s %s\n" "Qdrant:"             "qdrant-up, qdrant-down"
	@printf "  %-20s %s\n" "Indexer (bare):"     "indexer-local, query, query-rag"
	@printf "  %-20s %s\n" "Jobs (container):"   "export-run, indexer-run"
	@printf "  %-20s %s\n" "API (container):"    "api-build, api-up, api-down, api-logs, api-shell,"
	@printf "  %-20s %s\n" ""                    "api-ready, api-retrieve-smoke, api-ask-smoke"
	@printf "  %-20s %s\n" "Local stack:"        "stack-up, stack-down, stack-logs, stack-smoke"
	@printf "  %-20s %s\n" "UI:"                 "ui-build, ui-up, ui-down, ui-logs, ui-vps"
	@printf "  %-20s %s\n" "State:"              "status"
	@printf "  %-20s %s\n" "RunPod:"             "pod-up, pod-status, pod-stop, pod-terminate"
	@printf "  %-20s %s\n" "Droplet:"            "droplet-up, droplet-status, droplet-destroy, droplet-ssh, droplet-tunnel,"
	@printf "  %-20s %s\n" ""                    "droplet-sync-config, droplet-sync-data, remote-ghcr-login, remote-ghcr-logout"
	@printf "  %-20s %s\n" "VPS stack:"          "vps-export, vps-indexer, vps-up, vps-down   (run on the droplet, in /opt/devbuilder)"
	@printf "  %-20s %s\n" "Tests:"              "test, test-<module>  (e.g. test-auth)"
	@printf "  %-20s %s\n" "Cleanup:"            "clean, clean-all, teardown"

#### Setup ####

setup:
	uv sync --extra api --extra export --extra runpod --extra vps --group ui

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

#### Data pipeline ####

parse:
	uv run devbuilder --config $(CONFIG) parse

chunk:
	uv run devbuilder --config $(CONFIG) chunk

pipeline:
	uv run devbuilder --config $(CONFIG) pipeline

#### Model export ####

export-encoder:
	uv run devbuilder --config $(CONFIG) export-encoder --validate

export-reranker:
	uv run devbuilder --config $(CONFIG) export-reranker --validate

#### Qdrant (container) ####

qdrant-up:
	$(COMPOSE_LOCAL) up -d --wait qdrant

qdrant-down:
	$(COMPOSE_LOCAL) stop qdrant

#### Indexing & querying  ####

# Needs: pipeline, export-encoder, qdrant-up.
indexer-local:
	@$(LOAD_ENV) uv run devbuilder --config $(CONFIG) index --force-recreate

query:
	@$(LOAD_ENV) uv run devbuilder --config $(CONFIG) query --examples

query-rag: export VLLM_ENDPOINT = $(shell $(STATE_GET) vllm_endpoint)
query-rag:
	@$(LOAD_ENV) uv run devbuilder --config $(CONFIG) query --examples --with-decoder


# RunPod lifecycle

status:
	@$(DEPLOY) status

pod-up:
	$(DEPLOY) pod up

pod-status:
	$(DEPLOY) pod status

pod-stop:
	$(DEPLOY) pod stop

pod-terminate:
	$(DEPLOY) pod terminate


# API (container)

api-build:
	$(COMPOSE_LOCAL) build api

# Needs: export-encoder, export-reranker (artifacts are bind-mounted).
api-up: export VLLM_ENDPOINT = $(shell $(STATE_GET) vllm_endpoint)
api-up:
	$(COMPOSE_LOCAL) up -d --build --wait api

api-down:
	$(COMPOSE_LOCAL) stop api

api-logs:
	$(COMPOSE_LOCAL) logs -f api

api-shell:
	$(COMPOSE_LOCAL) exec api /bin/bash

api-ready:
	@$(LOAD_ENV) \
	curl -sS --fail-with-body http://localhost:8080/ready \
		-H "Authorization: Bearer $$$(TOKEN_VAR)" \
		| python -m json.tool

api-ask-smoke:
	@$(LOAD_ENV) \
	curl -sS --fail-with-body -X POST http://localhost:8080/ask \
		-H "Authorization: Bearer $$$(TOKEN_VAR)" \
		-H "Content-Type: application/json" \
		-d '{"query": "Who is the Hatter?"}' \
		| python -m json.tool

api-retrieve-smoke:
	@$(LOAD_ENV) \
	curl -sS --fail-with-body -X POST http://localhost:8080/retrieve \
		-H "Authorization: Bearer $$$(TOKEN_VAR)" \
		-H "Content-Type: application/json" \
		-d '{"query": "Who is the Hatter?"}' \
		| python -m json.tool


#### Local one-shots ####

indexer-run:
	$(COMPOSE_LOCAL) run --rm indexer

export-run:
	$(COMPOSE_LOCAL) run --rm export

#### Local backend stack ####

stack-up: export VLLM_ENDPOINT = $(shell $(STATE_GET) vllm_endpoint)
stack-up:
	$(COMPOSE_LOCAL) up -d --wait

stack-down:
	$(COMPOSE_LOCAL) down

stack-logs:
	$(COMPOSE_LOCAL) logs -f

stack-smoke:
	$(COMPOSE_LOCAL) ps --format '{{.Service}}\t{{.Status}}'
	@echo
	@curl -sS --fail-with-body http://localhost:8080/health && printf "\nAPI: ok\n"



# VPS lifecycle (run from the laptop)

droplet-up:
	$(DEPLOY) droplet up

droplet-status:
	$(DEPLOY) droplet status

droplet-sync-config:
	$(DEPLOY) droplet sync-config

droplet-sync-data:
	$(DEPLOY) droplet sync-data

# Targets that address the droplet resolve its IP from state when they run.
# Recursive (=), not simple (:=): a := here is evaluated at parse time on
# every make invocation, including on the droplet where uv does not exist.
droplet-ssh droplet-tunnel remote-ghcr-login remote-ghcr-logout: DROPLET_IP = $(shell $(STATE_GET) droplet_ip 2>/dev/null)

droplet-ssh:
	$(require_droplet)
	@ssh $(SSH_OPTS) root@$(DROPLET_IP)

droplet-tunnel:
	$(require_droplet)
	@ssh $(SSH_OPTS) -L 8080:127.0.0.1:8080 root@$(DROPLET_IP)

droplet-destroy:
	$(DEPLOY) droplet destroy

remote-ghcr-login:
	$(require_droplet)
	@$(LOAD_ENV) VPS_HOST=$(DROPLET_IP) ./scripts/remote_ghcr_login.sh

# Drop the stored registry credential once the images are pulled.
remote-ghcr-logout:
	$(require_droplet)
	@ssh root@$(DROPLET_IP) "docker logout ghcr.io"

# The vps-* targets run on the droplet, from the synced copy of this file in
# /opt/devbuilder: they pull from GHCR with the droplet's login and bind-mount
# the droplet's paths.
vps-export:
	$(require_vps)
	$(VPS_TAGS) $(COMPOSE_VPS) run --rm export devbuilder export-encoder --validate
	$(VPS_TAGS) $(COMPOSE_VPS) run --rm export devbuilder export-reranker --validate
	@$(LOAD_ENV) docker image rm ghcr.io/$$GHCR_OWNER/devbuilder-export:$(TAG) || true

vps-indexer:
	$(require_vps)
	$(VPS_TAGS) $(COMPOSE_VPS) run --rm indexer

vps-up:
	$(require_vps)
	$(VPS_TAGS) $(COMPOSE_VPS) up -d

vps-down:
	$(require_vps)
	$(VPS_TAGS) $(COMPOSE_VPS) down


#### UI (Gradio) ####

ui-build:
	$(COMPOSE) --profile local --profile ui build ui

ui-up: export VLLM_ENDPOINT = $(shell $(STATE_GET) vllm_endpoint)
ui-up: ui-build
	$(COMPOSE) --profile local --profile ui up -d --wait ui
	@echo "UI running at http://localhost:7860"

ui-down:
	$(COMPOSE) --profile local --profile ui stop ui

ui-logs:
	$(COMPOSE) --profile local --profile ui logs -f ui

ui-vps:
	@API_BASE_URL=http://localhost:8080 \
	 API_TOKEN=$$(grep '^VPS_API_TOKEN=' .env | cut -d= -f2-) \
	 uv run python -m devbuilder.ui.gradio_app

#### Tests ####

# Tests read no secrets, state, or remote service.
test:
	uv run pytest -q

# test-<module> runs tests/test_<module>.py; hyphens map to underscores.
test-%:
	uv run pytest -q tests/test_$(subst -,_,$*).py

#### Cleanup ####

clean:
	rm -rf data/processed/* artifacts/models/*
	$(COMPOSE_LOCAL) down -v

clean-all: clean pod-terminate

teardown: pod-terminate droplet-destroy
