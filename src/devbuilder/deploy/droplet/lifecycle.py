"""
Droplet lifecycle (DigitalOcean): sequence the provider operations and
record droplet and firewall in the deployment state. Provider calls live in
``manager`` and ``firewall``.
"""

from __future__ import annotations

import logging

from devbuilder.deploy.droplet.config import DropletConfig
from devbuilder.deploy.droplet.firewall import delete_firewall, ensure_firewall, resolve_ssh_cidr
from devbuilder.deploy.droplet.manager import (
    create_droplet,
    destroy_droplet,
    find_droplet_by_name,
    get_status,
    smoke_test_ssh,
    wait_until_ready,
)
from devbuilder.deploy.state import clear_state, update_state

logger = logging.getLogger(__name__)

STATE_FIELDS = ("droplet_id", "droplet_ip", "firewall_id")


def _firewall_name(config: DropletConfig) -> str:
    return f"{config.name}-fw"


def _public_ip(status: dict) -> str | None:
    return next(
        (
            network["ip_address"]
            for network in status.get("networks", {}).get("v4", [])
            if network.get("type") == "public"
        ),
        None,
    )


def up(config: DropletConfig) -> None:
    """
    Create or reuse the droplet, attach the firewall, then wait for readiness
    and check SSH. Order matters: the SSH source CIDR is resolved before
    anything is billed, and the firewall is attached as soon as the droplet id
    exists, and a changed operator IP is repaired before SSH is attempted.
    The droplet is reached by IP only; it never enters public DNS. State is
    written as each resource appears, so a failure midway leaves what exists
    recorded for retry or teardown.
    """
    ssh_source_cidr = resolve_ssh_cidr()

    droplet_id, _ = create_droplet(config)
    update_state(droplet_id=droplet_id)

    firewall_id = ensure_firewall(
        name=_firewall_name(config),
        droplet_id=droplet_id,
        ssh_source_cidr=ssh_source_cidr,
    )
    update_state(firewall_id=firewall_id)

    ip = wait_until_ready(droplet_id)
    update_state(droplet_ip=ip)

    smoke_test_ssh(ip)
    logger.info("Droplet %s is up at %s", droplet_id, ip)


def status(config: DropletConfig) -> None:
    """
    Report droplet state as DigitalOcean sees it.
    """
    droplet = find_droplet_by_name(config.name)
    if droplet is None:
        logger.info("No droplet named %r found", config.name)
        return

    current = get_status(droplet["id"]) or {}
    logger.info(
        "Droplet %r: id=%s, status=%s, ip=%s",
        config.name,
        droplet["id"],
        current.get("status"),
        _public_ip(current),
    )


def destroy(config: DropletConfig) -> None:
    """
    Delete the droplet and its firewall, then forget both. Every step is a
    no-op when its resource is already gone, so a partial teardown converges
    on re-run.
    """
    droplet = find_droplet_by_name(config.name)
    if droplet is not None:
        destroy_droplet(droplet["id"])
    else:
        logger.info("No droplet named %r found; nothing to destroy", config.name)

    delete_firewall(_firewall_name(config))
    clear_state(*STATE_FIELDS)
