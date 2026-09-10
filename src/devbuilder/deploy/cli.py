"""
Operator entry point for provisioning: ``devbuilder-deploy``.

One process, one ``.env`` load, one state file. Provider SDKs are imported
per command so ``status`` and ``get`` work without every extra installed.

Exit codes: 1 generic provider error, 2 auth, 3 quota, 4 not ready,
5 config, 6 state file unreadable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, replace

from dotenv import load_dotenv

import devbuilder.config as cfg
from devbuilder.deploy.state import DeployState, StateError, load_state
from devbuilder.paths import paths

logger = logging.getLogger("devbuilder-deploy")

EXIT_ERROR, EXIT_AUTH, EXIT_QUOTA, EXIT_NOT_READY, EXIT_CONFIG, EXIT_STATE = 1, 2, 3, 4, 5, 6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="devbuilder-deploy")
    parser.add_argument(
        "--config",
        default=str(paths.configs / "settings.yaml"),
        help="Path to YAML config file",
    )
    top = parser.add_subparsers(dest="target", required=True)

    top.add_parser("status", help="Show the deployment state file")

    get = top.add_parser("get", help="Print one state field (empty if unset)")
    get.add_argument("field", choices=DeployState.field_names())

    pod = top.add_parser("pod", help="vLLM pod on RunPod").add_subparsers(
        dest="command", required=True
    )
    pod.add_parser("up", help="Create or resume the pod, wait, run smoke tests")
    pod.add_parser("status", help="Show pod status")
    pod.add_parser("stop", help="Stop the pod (releases GPU, keeps volume and endpoint)")
    pod.add_parser("terminate", help="Delete pod and volume")

    droplet = top.add_parser("droplet", help="VPS on DigitalOcean").add_subparsers(
        dest="command", required=True
    )
    droplet.add_parser("up", help="Create droplet, check readiness, firewall, and SSH")
    droplet.add_parser("status", help="Show droplet status")
    droplet.add_parser("destroy", help="Delete droplet and its firewall")
    droplet.add_parser("sync-config", help="Push compose files, Makefile, and a rendered .env")
    droplet.add_parser("sync-data", help="Push the raw corpus")
    return parser


def _require_droplet_ip(state: DeployState) -> str:
    if not state.droplet_ip:
        raise SystemExit("No droplet in state; run `devbuilder-deploy droplet up` first.")
    return state.droplet_ip


def _run_pod(args: argparse.Namespace, settings: dict) -> int:
    from devbuilder.deploy.pod import (
        PodAuthError,
        PodConfig,
        PodError,
        PodNotReadyError,
        PodQuotaError,
        lifecycle,
    )

    config = PodConfig(
        model_id=settings["decoder"]["model_id"],
        vllm_port=settings["serving"]["vllm_port"],
        **settings["runpod"],
    )
    handlers = {
        "up": lifecycle.up,
        "status": lifecycle.status,
        "stop": lifecycle.stop,
        "terminate": lifecycle.terminate,
    }
    try:
        handlers[args.command](config)
    except PodAuthError as exc:
        logger.error("Auth error: %s. Check RUNPOD_API_KEY in .env", exc)
        return EXIT_AUTH
    except PodQuotaError as exc:
        logger.error("Quota error: %s. Try a different gpu_type in settings.yaml", exc)
        return EXIT_QUOTA
    except PodNotReadyError as exc:
        logger.error("Pod not ready: %s", exc)
        return EXIT_NOT_READY
    except PodError as exc:
        logger.error("RunPod error: %s", exc)
        return EXIT_ERROR
    return 0


def _run_droplet(args: argparse.Namespace, settings: dict) -> int:
    from devbuilder.deploy.droplet import (
        DropletAuthError,
        DropletConfig,
        DropletConfigError,
        DropletError,
        DropletNotReadyError,
        DropletQuotaError,
        lifecycle,
        sync,
    )

    do_settings = dict(settings["vps"]["digitalocean"])
    if "tags" in do_settings:
        do_settings["tags"] = tuple(do_settings["tags"])
    config = DropletConfig(**do_settings)

    try:
        if args.command == "sync-config":
            state = load_state()
            sync.sync_config(_require_droplet_ip(state), state)
        elif args.command == "sync-data":
            sync.sync_data(_require_droplet_ip(load_state()))
        elif args.command == "up":
            # Creation inputs are validated here and nowhere else: status and
            # destroy must keep working when the bootstrap file or the SSH
            # fingerprint is missing, or a billable droplet could not be removed.
            if not config.ssh_key_fingerprint:
                raise DropletConfigError("DO_SSH_KEY_FINGERPRINT is not set in .env")
            init_path = paths.configs / "droplet_init.yaml"
            if not init_path.is_file():
                raise DropletConfigError(f"cloud-init file not found: {init_path}")
            lifecycle.up(replace(config, user_data=init_path.read_text()))
        elif args.command == "status":
            lifecycle.status(config)
        else:
            lifecycle.destroy(config)
    except sync.SyncError as exc:
        logger.error("Sync failed: %s", exc)
        return EXIT_ERROR
    except DropletAuthError as exc:
        logger.error("Auth error: %s. Check DIGITALOCEAN_TOKEN in .env", exc)
        return EXIT_AUTH
    except DropletQuotaError as exc:
        logger.error("Quota error: %s. Try a different size/region in settings.yaml", exc)
        return EXIT_QUOTA
    except DropletNotReadyError as exc:
        logger.error("Droplet not ready: %s", exc)
        return EXIT_NOT_READY
    except DropletConfigError as exc:
        logger.error("Config error: %s", exc)
        return EXIT_CONFIG
    except DropletError as exc:
        logger.error("VPS error: %s", exc)
        return EXIT_ERROR
    return 0


def main() -> int:
    args = _parser().parse_args()

    # `get` is consumed by Make; keep its stdout to the bare value.
    if args.target == "get":
        try:
            value = getattr(load_state(), args.field)
        except StateError as exc:
            print(exc, file=sys.stderr)
            return EXIT_STATE
        print("" if value is None else value)
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Readiness polling would otherwise print one httpx line per probe, and
    # pydo (azure.core) logs every request and response at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    # Exported environment variables take precedence over .env values.
    load_dotenv(paths.base_dir / ".env", override=False)

    if args.target == "status":
        try:
            state = load_state()
        except StateError as exc:
            logger.error("%s", exc)
            return EXIT_STATE
        for key, value in asdict(state).items():
            print(f"{key:15} {value if value is not None else '-'}")
        return 0

    settings = cfg.load_config(args.config)
    if args.target == "pod":
        return _run_pod(args, settings)
    return _run_droplet(args, settings)


if __name__ == "__main__":
    sys.exit(main())
