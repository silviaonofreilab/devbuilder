"""
Push deployment inputs to the droplet: compose files, Makefile, a rendered
``.env``, and the raw corpus.

The droplet's ``.env`` is derived, not copied: only an allowlist of keys is
forwarded, the production API token is renamed into place, and
the decoder endpoint comes from the deployment state rather than from the
operator's file.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from devbuilder.deploy.state import DeployState
from devbuilder.paths import paths

logger = logging.getLogger(__name__)

REMOTE_USER = "root"
REMOTE_DIR = "/opt/devbuilder"
REMOTE_DATA_DIR = f"{REMOTE_DIR}/data/raw"

# Files copied verbatim to the droplet.
CONFIG_FILES = ("compose.yaml", "compose.vps.yaml", "Makefile")

# Keys renamed on the droplet: local name -> remote name.
RENAMED_KEYS = {"VPS_API_TOKEN": "API_TOKEN"}

# The only assignments copied to the droplet. An allowlist, so a credential
# added to .env later is not forwarded until someone decides it should be.
# Everything else in .env (provisioning tokens, SSH details, the dev API
# token, GHCR login) stays on the laptop.
DROPLET_KEYS = frozenset(
    {
        "VPS_API_TOKEN",  # renamed to API_TOKEN below
        "VLLM_API_KEY",
        "TUNNEL_TOKEN",
        "CLOUDFLARE_TAG",
        "GHCR_OWNER",
        "HF_TOKEN",
    }
)


class SyncError(Exception):
    """A sync precondition failed or a remote command did not succeed."""


def render_env(env_path: Path, state: DeployState) -> str:
    """
    Produce the droplet's ``.env`` text from the operator's file and the state.
    """
    if not env_path.is_file():
        raise SyncError(f"No .env found at {env_path}")

    kept: list[str] = []
    for line in env_path.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            kept.append(line)  # comments and blank lines
            continue
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip()
        if key not in DROPLET_KEYS:
            continue
        _, _, value = line.partition("=")
        kept.append(f"{RENAMED_KEYS.get(key, key)}={value}")

    kept.append("")
    kept.append("# Written by devbuilder-deploy from the deployment state.")
    kept.append(f"VLLM_ENDPOINT={state.vllm_endpoint or ''}")
    return "\n".join(kept) + "\n"


def _config_paths() -> list[Path]:
    found: list[Path] = []
    for name in CONFIG_FILES:
        path = paths.base_dir / name
        if not path.is_file():
            raise SyncError(f"Config file not found: {path}")
        found.append(path)
    return found


def _run(args: list[str], **kwargs) -> None:
    try:
        subprocess.run(args, check=True, **kwargs)
    except FileNotFoundError as exc:
        raise SyncError(f"Required command not found: {exc.filename}") from exc
    except subprocess.CalledProcessError as exc:
        raise SyncError(f"{args[0]} failed (exit {exc.returncode})") from exc


def sync_config(host: str, state: DeployState, env_path: Path | None = None) -> None:
    """
    Create the remote dir, scp the config files, and write a mode-600 ``.env``.
    """
    env_body = render_env(env_path or paths.base_dir / ".env", state)
    config_files = _config_paths()
    target = f"{REMOTE_USER}@{host}"

    logger.info("Ensuring remote dir %s:%s", target, REMOTE_DIR)
    _run(["ssh", target, f"mkdir -p {REMOTE_DIR}"])

    logger.info("Copying config files: %s", ", ".join(p.name for p in config_files))
    _run(["scp", *[str(p) for p in config_files], f"{target}:{REMOTE_DIR}/"])

    # Stream the .env over stdin so secrets never touch local disk;
    # umask 077 creates it 600 with no world-readable window.
    logger.info("Writing rendered .env to %s:%s/.env", target, REMOTE_DIR)
    _run(
        ["ssh", target, f"umask 077 && cat > {REMOTE_DIR}/.env && chmod 600 {REMOTE_DIR}/.env"],
        input=env_body,
        text=True,
    )
    logger.info("Config sync complete.")


def sync_data(host: str, local_dir: Path | None = None) -> None:
    """
    Create the remote data dir and rsync the raw corpus into it.

    Placeholder for a production-realistic corpus-delivery approach.
    """
    raw = local_dir or paths.raw
    if not raw.is_dir() or not any(raw.iterdir()):
        raise SyncError(f"No corpus found at {raw}; nothing to sync.")

    target = f"{REMOTE_USER}@{host}"
    remote_spec = f"{target}:{REMOTE_DATA_DIR}/"
    source_spec = f"{str(raw).rstrip('/')}/"

    logger.info("Ensuring remote dir %s:%s", target, REMOTE_DATA_DIR)
    _run(["ssh", target, f"mkdir -p {REMOTE_DATA_DIR}"])

    logger.info("Syncing %s -> %s", source_spec, remote_spec)
    _run(["rsync", "-avz", source_spec, remote_spec])
    logger.info("Corpus sync complete.")
