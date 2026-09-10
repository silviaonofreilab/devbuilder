from __future__ import annotations

import logging
import os
import subprocess
import time

import pydo
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from devbuilder.deploy.droplet.config import DropletConfig

logger = logging.getLogger(__name__)

# Exceptions


class DropletError(Exception):
    """
    Base for all VPS operational errors.
    """


class DropletAuthError(DropletError):
    """
    Invalid or missing DIGITALOCEAN_TOKEN.
    """


class DropletQuotaError(DropletError):
    """
    Droplet limit reached or size unavailable in region.
    """


class DropletNotReadyError(DropletError):
    """
    Droplet did not become active or reachable within timeout.
    """


def _translate_pydo_error(exc: Exception) -> DropletError:
    """
    Map pydo / HTTP exceptions to typed DropletError variants.
    """
    msg = str(exc).lower()
    if "401" in msg or "unauthorized" in msg or "authenticat" in msg:
        return DropletAuthError("DIGITALOCEAN_TOKEN rejected by DigitalOcean API")
    if "402" in msg or "limit" in msg or "quota" in msg or "not available" in msg:
        return DropletQuotaError(str(exc))
    return DropletError(str(exc))


# Credentials


_client_instance: pydo.Client | None = None


def _require_token() -> str:
    token = os.getenv("DIGITALOCEAN_TOKEN")
    if not token:
        raise DropletAuthError("DIGITALOCEAN_TOKEN not set in environment")
    return token


def get_client() -> pydo.Client:
    """
    Return the pydo client, creating it on first use.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = pydo.Client(token=_require_token())
    return _client_instance


# Helpers


def find_droplet_by_name(name: str) -> dict | None:
    """
    Return the droplet dict matching `name`, or None if absent.
    """
    try:
        resp = get_client().droplets.list()
    except Exception as exc:
        raise _translate_pydo_error(exc) from exc

    for droplet in resp.get("droplets", []):
        if droplet.get("name") == name:
            return droplet
    return None


def _public_ipv4(droplet: dict) -> str | None:
    """
    Extract the public IPv4 address from a droplet payload, if assigned.
    """
    for net in droplet.get("networks", {}).get("v4", []):
        if net.get("type") == "public":
            return net.get("ip_address")
    return None


# Lifecycle


def _is_not_found(exc: Exception) -> bool:
    """
    True if the exception represents a 404 from the DigitalOcean API.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 404
    return "404" in str(exc) or "not found" in str(exc).lower()


def get_status(droplet_id: int) -> dict | None:
    """
    Return droplet info dict, or None if the droplet no longer exists.
    """
    try:
        resp = get_client().droplets.get(droplet_id=droplet_id)
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise _translate_pydo_error(exc) from exc
    return resp.get("droplet")


def create_droplet(config: DropletConfig) -> tuple[int, str | None]:
    """
    Create a droplet (or return existing one with same name).

    Returns (droplet_id, public_ipv4). The IP may be None immediately after
    creation; call wait_until_ready to poll until it is assigned.
    """
    existing = find_droplet_by_name(config.name)
    if existing is not None:
        ip = _public_ipv4(existing)
        logger.info(
            "Droplet %r already exists (id=%s, ip=%s); reusing",
            config.name,
            existing["id"],
            ip,
        )
        return existing["id"], ip

    logger.info(
        "Creating droplet %r (size=%s, region=%s, image=%s)",
        config.name,
        config.size,
        config.region,
        config.image,
    )

    body = {
        "name": config.name,
        "region": config.region,
        "size": config.size,
        "image": config.image,
        "ssh_keys": [config.ssh_key_fingerprint],
        "tags": list(config.tags),
        "ipv6": False,
        "monitoring": True,
        "backups": False,
    }
    if config.user_data:
        body["user_data"] = config.user_data

    try:
        resp = get_client().droplets.create(body=body)
    except Exception as exc:
        raise _translate_pydo_error(exc) from exc

    droplet = resp["droplet"]
    ip = _public_ipv4(droplet)
    logger.info("Droplet created (id=%s, ip=%s)", droplet["id"], ip)
    return droplet["id"], ip


def wait_until_ready(droplet_id: int, timeout: int = 600, interval: int = 5) -> str:
    """
    Poll until droplet is `active` and has a public IPv4, then probe SSH port.

    Returns the public IPv4 address. Raises DropletNotReadyError on timeout.
    """
    logger.info("Waiting for droplet %s to become active (timeout=%ds)", droplet_id, timeout)
    deadline = time.monotonic() + timeout

    ip: str | None = None
    while time.monotonic() < deadline:
        droplet = get_status(droplet_id)
        if droplet is not None and droplet.get("status") == "active":
            ip = _public_ipv4(droplet)
            if ip:
                break
        time.sleep(interval)

    if ip is None:
        raise DropletNotReadyError(
            f"Droplet {droplet_id} did not become active with a public IP within {timeout}s"
        )

    logger.info("Droplet active at %s; waiting for SSH", ip)
    _wait_ssh(ip, deadline)
    _wait_cloud_init(ip, deadline)
    logger.info("Droplet %s ready (ip=%s)", droplet_id, ip)
    return ip


@retry(
    retry=retry_if_exception_type(OSError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _probe_ssh(ip: str, port: int = 22, timeout: float = 5.0) -> None:
    """
    Open a TCP connection to the SSH port; raise OSError on failure.
    """
    import socket

    with socket.create_connection((ip, port), timeout=timeout):
        return


def _wait_ssh(ip: str, deadline: float, interval: int = 5) -> None:
    """
    Poll the SSH port until reachable or deadline expires.
    """
    while time.monotonic() < deadline:
        try:
            _probe_ssh(ip)
            return
        except OSError as exc:
            logger.debug("SSH probe to %s failed: %s", ip, exc)
        time.sleep(interval)

    raise DropletNotReadyError(f"SSH on {ip}:22 not reachable before timeout")


def smoke_test_ssh(ip: str) -> None:
    """
    Assert the SSH port accepts connections.
    """
    try:
        _probe_ssh(ip)
    except OSError as exc:
        raise DropletNotReadyError(f"SSH probe failed: {exc}") from exc
    logger.info("SSH smoke test passed (ip=%s)", ip)


def _wait_cloud_init(ip: str, deadline: float) -> None:
    """
    Block until cloud-init finishes on the droplet, so Docker/make are installed
    before we declare readiness.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DropletNotReadyError(f"No time left to wait for cloud-init on {ip}")

    target = f"root@{ip}"
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        target,
        "cloud-init status --wait",
    ]
    logger.info("Waiting for cloud-init to finish on %s", ip)
    try:
        result = subprocess.run(cmd, timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        raise DropletNotReadyError(f"cloud-init did not finish on {ip} within timeout") from exc

    if result.returncode != 0:
        raise DropletNotReadyError(
            f"cloud-init did not complete cleanly on {ip} (exit {result.returncode})"
        )
    logger.info("cloud-init complete on %s", ip)


def destroy_droplet(droplet_id: int) -> None:
    """
    Permanently delete the droplet. No-op if absent.
    """
    if get_status(droplet_id) is None:
        logger.info("destroy_droplet: %s not found; nothing to do", droplet_id)
        return

    try:
        get_client().droplets.destroy(droplet_id=droplet_id)
    except Exception as exc:
        raise _translate_pydo_error(exc) from exc

    logger.info("Droplet %s destroyed", droplet_id)
