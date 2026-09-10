#!/usr/bin/env bash
#
# Log the droplet into GHCR so it can pull private images.
# Reads GHCR_USER / GHCR_PAT from the environment (sourced from .env by the
# Makefile target) and VPS_HOST (the droplet IP, passed by Make from the
# deployment state). The PAT travels over SSH stdin, never as a visible
# argument. `docker login` does keep it (base64, mode 600) in
# /root/.docker/config.json until `make remote-ghcr-logout`.
set -euo pipefail

: "${VPS_HOST:?Missing VPS_HOST (run 'make droplet-up' first)}"
: "${GHCR_USER:?Missing GHCR_USER in .env}"
: "${GHCR_PAT:?Missing GHCR_PAT in .env}"
VPS_USER="${VPS_USER:-root}"

host="${VPS_HOST}"

echo "$GHCR_PAT" | ssh \
  -o StrictHostKeyChecking=accept-new \
  "${VPS_USER}@${host}" \
  "docker login ghcr.io -u ${GHCR_USER} --password-stdin"
