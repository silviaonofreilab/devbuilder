# DevBuilder

*DevBuilder* is a reference implementation for deploying a RAG system with ONNX Runtime, Qdrant, and vLLM. The encoder and reranker run on a DigitalOcean CPU droplet; the decoder runs on a RunPod GPU pod. A Gradio interface is accessible through Cloudflare Tunnel and Access.

The project covers model export and validation, container images, infrastructure provisioning, and deployment checks. It uses a small corpus, Lewis Carroll's *Alice's Adventures in Wonderland*, to keep the focus on deployment.

Resources are provisioned and managed from a local checkout using CLI commands and Make targets. The workflow supports on-demand deployment for development and demonstrations.

**Scope.** Automatic deployment through CI/CD, managed secrets, and monitoring are outside the current scope.

## Architecture

```
 internet        DigitalOcean droplet (CPU)                             RunPod pod (GPU)
┌────────────┐   ┌──────────────────────────────────────────────────┐   ┌──────────────────┐
│browser     │   │cloudflared ─► Gradio UI ─► FastAPI ─┬─► encoder  │   │                  │
│   │        │   │(outbound      :7860        :8080    ├─► Qdrant   │   │vLLM decoder      │
│   ▼        │   │ tunnel)                             ├─► reranker │   │:8000 (+ key)     │
│Cloudflare  ├──►│                                     └────────────┼──►│                  │
│Access +    │   │                                  HTTPS + API key │   │                  │
│rate limit  │   │                                                  │   │                  │
│            │   │       encoder, reranker: ONNX Runtime on CPU     │   │                  │
└────────────┘   └──────────────────────────────────────────────────┘   └──────────────────┘
```

- **Encoder** — a BERT-family model exported to ONNX (Optimum), graph-optimized (O3) and quantized using ONNX Runtime, served on CPU, with pooling done in NumPy.
- **Retrieval** — Qdrant holds chunk embeddings; the API fetches `fetch_k` candidates and a cross-encoder **reranker** (also ONNX) keeps the best `top_k`.
- **Decoder** — an instruction-tuned LLM served by vLLM on a single-GPU RunPod pod, reached through RunPod's HTTPS proxy with an API key.
- **Ingress** — the UI is built with Gradio, published through a Cloudflare Tunnel and gated by Cloudflare Access (email one-time PIN) and a rate-limit rule. The API port is published only on the droplet's localhost interface.

## Stack

FastAPI · Qdrant · ONNX Runtime + Optimum · vLLM · Gradio · Docker Compose · GitHub Actions + GHCR · DigitalOcean · RunPod · Cloudflare Tunnel + Access · `uv`

## Layout

```
src/devbuilder/
  api.py            FastAPI app: /health /ready /retrieve /ask
  cli.py            `devbuilder`: parse, chunk, index, query, export-encoder, export-reranker
  deploy/           `devbuilder-deploy` (cli.py), deployment state (state.py),
    pod/            RunPod GPU pod: config, manager, lifecycle
    droplet/        DigitalOcean droplet: config, manager, firewall, lifecycle, sync
  manifest.py       artifact contract between export and runtime
  export/           ONNX export, optimize, quantize, validate
  encoder/ reranker/ decoder/ vectorstore/ rag.py   runtime components
  pipelines/        corpus parsing and chunking
  observability.py  request ids and access log
docker/             one Dockerfile per image: api, export, indexer, ui
compose.yaml        base stack (local); compose.vps.yaml  VPS overlay (GHCR images, mounts, cloudflared)
configs/            settings.yaml (models, chunking, prompts, pod and droplet specs), droplet_init.yaml (cloud-init)
scripts/            GHCR login helper
tests/              unit, artifact and, integration tests
Makefile            operator entry point (`make help`)
```

## Prerequisites

Local: Python ≥ 3.11, `uv`, Docker Desktop, `ssh`/`scp`/`rsync`.

Accounts: DigitalOcean (with a registered SSH key), RunPod (with secrets `vllm_api_key` and, for gated models, `hf_token`), GitHub (Actions + GHCR), Cloudflare (a zone for the public hostname; Zero Trust free plan).

## Credentials

Copy `.env.example` to `.env`. It is operator input, and its keys fall in two groups:

| Not included in droplet's `.env` | Shipped to the droplet |
|---|---|
| `DIGITALOCEAN_TOKEN`, `DO_SSH_KEY_FINGERPRINT`, `SSH_ALLOW_CIDR` | `VLLM_API_KEY` |
| `RUNPOD_API_KEY` | `VPS_API_TOKEN` (becomes `API_TOKEN` there) |
| `GHCR_USER`, `GHCR_PAT` (classic PAT, `read:packages` only) | `GHCR_OWNER`, `CLOUDFLARE_TAG`, `TUNNEL_TOKEN` |
| `API_TOKEN` (dev token for the local stack) | `HF_TOKEN` (only if set; gated models only) |

The droplet's `.env` is generated from the variables in the right-hand column, with the decoder endpoint added from the deployment state. Each container receives only the environment variables specified for its service in Compose.

## Configuration

`configs/settings.yaml` contains application and provisioning settings, including model selection, chunking, prompts, and pod and droplet specifications.

## Model artifacts

Every exported model includes a `manifest.json` recording model identity, file checksums, export settings, and runtime metadata. Exports use staging, with the previous artifact retained until replacement succeeds.

## Deployment state

Deployment commands record resource IDs, the droplet IP, and the decoder endpoint in `.devbuilder/state.json` (gitignored). Run `make status` to view the current deployment state.

## Quick start (local)

```bash
make setup                                # uv sync, all extras
make pipeline                             # parse + chunk the corpus
make export-encoder export-reranker       # ONNX artifacts under artifacts/models/
make qdrant-up && make indexer-run        # embed the corpus into Qdrant
make stack-up                             # api (+ ui) in Docker
make api-ready                            # qdrant_ok true, llm_ok false — retrieval works without a GPU
make api-retrieve-smoke
```

Add the decoder:

```bash
make pod-up                               # create or resume the pod, wait, smoke-test, record the endpoint
make api-up                               # restarts the api with the endpoint from state
make api-ask-smoke                        # full RAG answer
make ui-up                                # Gradio at http://localhost:7860
```

End session::

```bash
# Choose one:
make pod-stop                             # stop up for later use
make pod-terminate                        # permanently delete the pod
```

## Images

Images are built by the `docker` workflow on a version tag and pushed to `ghcr.io/<owner>/devbuilder-{api,export,indexer,ui}:<version>` for linux/amd64. Each image installs only what it needs (the export image is the only one with torch; the UI image has gradio and httpx), runs as an unprivileged fixed UID, and pins its base images by digest.

```bash
# bump `version` in pyproject.toml, TAG in the Makefile, project.version in settings.yaml
uv lock && git commit -am "Bump to X.Y.Z"
git tag vX.Y.Z && git push origin main vX.Y.Z
```

## Deploy to the VPS

### Cloudflare (one-time, in the dashboard)

1. Zero Trust → Networking → Tunnels → create a tunnel (Cloudflared); copy the token into `.env` as `TUNNEL_TOKEN`. Add a route: subdomain of your zone → `http://ui:7860`. This creates the CNAME.
2. Zero Trust → Integrations → Identity providers → add One-time PIN.
3. Zero Trust → Access controls → Applications → self-hosted, that hostname; policy Allow with your email(s); instant authentication on.
4. Zone → Security → Security rules → Rate limiting rule for the hostname.

Deleting a tunnel leaves its CNAME behind; point the record at the new tunnel id if you recreate one.

### Deployment details

*droplet* refers to the DigitalOcean machine and its lifecycle; *vps* refers to the application stack deployed on that machine.

Run provisioning commands from your laptop. After connecting with `make droplet-ssh`, run the `vps-*` targets from `/opt/devbuilder` on the droplet.

Images are pulled on every `vps-up` (`pull_policy: always`). For private images, run `make remote-ghcr-login` if you have logged out or the saved credentials are no longer valid.


```bash
# laptop
make pod-up                               # if not already running
make droplet-up                           # create droplet, cloud-init (Docker, upgrades), firewall (SSH only, from your IP)
make droplet-sync-config                  # compose files, Makefile, rendered .env
make droplet-sync-data                    # corpus
make remote-ghcr-login                    # if the packages are private

make droplet-ssh
  # droplet
  cd /opt/devbuilder
  make vps-export                         # one-shot: export both models on the box
  make vps-indexer                        # one-shot: embed the corpus
  make vps-up                             # qdrant, api, ui, cloudflared
  exit
# back on laptop
make remote-ghcr-logout                   # optional: drop the registry credential from the droplet
```

### Verify

```bash
make droplet-tunnel                       # terminal 1: forwards localhost:8080 to the droplet's API
make api-ready TARGET=vps                 # terminal 2: uses VPS_API_TOKEN
make api-ask-smoke TARGET=vps
```

Open your public hostname in a private browser window. Confirm that Cloudflare Access requires authentication before displaying the Gradio UI.

### Teardown

```bash
make pod-terminate                        # or pod-stop to keep the volume
make droplet-destroy                      # droplet and its firewall
make status                               # inspect deployment status
make clean                                # local processed data, artifacts, Qdrant volume
```

## Tests

Run `make test` to run the test suite. Integration tests download and export models; artifact tests require locally exported models and skip when those files are absent.

During `make pod-up`, deployment checks verify pod health, an authenticated completion, and rejection of requests without an API key.


## Security notes

- The public UI is protected by Cloudflare Access. The API port is published only on the droplet’s localhost interface, and SSH access is restricted to the configured CIDR.
- API requests require a bearer token, except for `/health` and the documentation routes. The RunPod decoder endpoint is publicly reachable and protected by a separate API key.
- Qdrant has no API key and publishes no host ports on the VPS; services access it through the Docker network.
- Secrets are stored in local `.env` files, with selected variables forwarded to the droplet. SSH uses root access with key authentication.


## Swapping models and data

- **Encoder**: Update the encoder settings in `configs/settings.yaml`, then re-export and re-index. Compatible models must provide `last_hidden_state` for runtime pooling.
- **Reranker**: Compatible sequence-classification cross-encoder with a single-logit head. Update the reranker settings in `configs/settings.yaml`, then re-export.
- **Decoder**: Use a compatible model served by vLLM through the OpenAI chat API. Update the decoder and RunPod settings in `configs/settings.yaml` to match the model and its resource requirements.
- **Corpus**: Adapt `pipelines/preprocessing.py::parse_text` for the new corpus, then re-run preprocessing and indexing. The current parser handles the chapter headings and ending of *Alice’s Adventures in Wonderland*.

For VPS deployments, changes to `configs/settings.yaml` require rebuilding and publishing the affected application images. Restart the API after replacing encoder or reranker artifacts. Changes to the decoder model or pod launch settings require recreating the pod; `make pod-up` reuses an existing pod without applying those changes.

Only the default models have been tested through the complete pipeline.


## Troubleshooting

- **API returns 401 through the SSH tunnel:** Check that you used `TARGET=vps` so the command uses `VPS_API_TOKEN`.
- **GHCR returns `unauthorized`:** Run `make remote-ghcr-login` from your laptop, then retry the deployment command on the droplet.
- **Cloudflare error 1033:** Check that `cloudflared` is running on the droplet and that the hostname’s DNS record points to the correct tunnel.
- **`ManifestSchemaError` when loading a model:** If the artifact uses an older manifest schema, re-export the model.


## License

MIT — see [LICENSE](LICENSE).
