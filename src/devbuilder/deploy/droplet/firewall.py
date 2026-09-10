from __future__ import annotations

import ipaddress
import logging
import os

import httpx

from devbuilder.deploy.droplet.manager import DropletError, _translate_pydo_error, get_client

logger = logging.getLogger(__name__)


class DropletConfigError(DropletError):
    """Missing or invalid VPS-related env config."""


def resolve_ssh_cidr() -> str:
    """
    Resolve the SSH source CIDR from env.
    """
    val = os.getenv("SSH_ALLOW_CIDR", "").strip()
    if not val:
        raise DropletConfigError(
            "SSH_ALLOW_CIDR is not set. Set it to a CIDR (e.g. 1.2.3.4/32) "
            "or 'auto' to detect your current public IP."
        )
    if val == "auto":
        try:
            resp = httpx.get("https://api.ipify.org", timeout=5)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise DropletConfigError(f"Could not detect public IP: {exc}") from exc

        ip = resp.text.strip()
        try:
            ipaddress.ip_address(ip)
        except ValueError as exc:
            raise DropletConfigError(
                f"Public IP lookup returned an invalid address: {ip!r}"
            ) from exc

        return f"{ip}/32"
    return val


def _find_firewall_by_name(name: str) -> dict | None:
    try:
        resp = get_client().firewalls.list()
    except Exception as exc:
        raise _translate_pydo_error(exc) from exc

    for fw in resp.get("firewalls", []):
        if fw.get("name") == name:
            return fw
    return None


def ensure_firewall(
    name: str,
    droplet_id: int,
    ssh_source_cidr: str,
    allow_http: bool = False,  # True only if not using Cloudflare Tunnel
) -> str:
    """
    Create or update a DigitalOcean Cloud Firewall attached to the droplet.
    Returns firewall_id.
    """
    inbound = [
        {"protocol": "tcp", "ports": "22", "sources": {"addresses": [ssh_source_cidr]}},
    ]
    if allow_http:
        inbound += [
            {"protocol": "tcp", "ports": "80", "sources": {"addresses": ["0.0.0.0/0", "::/0"]}},
            {"protocol": "tcp", "ports": "443", "sources": {"addresses": ["0.0.0.0/0", "::/0"]}},
        ]

    outbound = [
        {"protocol": "tcp", "ports": "all", "destinations": {"addresses": ["0.0.0.0/0", "::/0"]}},
        {"protocol": "udp", "ports": "all", "destinations": {"addresses": ["0.0.0.0/0", "::/0"]}},
        {"protocol": "icmp", "destinations": {"addresses": ["0.0.0.0/0", "::/0"]}},
    ]

    # droplet_ids replaces the full attachment list on update; this firewall
    # is assumed to manage exactly one droplet.
    body = {
        "name": name,
        "inbound_rules": inbound,
        "outbound_rules": outbound,
        "droplet_ids": [droplet_id],
    }

    existing = _find_firewall_by_name(name)
    try:
        if existing:
            get_client().firewalls.update(firewall_id=existing["id"], body=body)
            logger.info("Firewall %r updated (id=%s)", name, existing["id"])
            return existing["id"]
        resp = get_client().firewalls.create(body=body)
        fw_id = resp["firewall"]["id"]
        logger.info("Firewall %r created (id=%s)", name, fw_id)
        return fw_id
    except Exception as exc:
        raise _translate_pydo_error(exc) from exc


def delete_firewall(name: str) -> bool:
    """
    Delete the named firewall if it exists. Returns True if one was removed.
    """
    existing = _find_firewall_by_name(name)
    if existing is None:
        logger.info("Firewall %r not found; nothing to delete", name)
        return False

    try:
        get_client().firewalls.delete(firewall_id=existing["id"])
    except Exception as exc:
        raise _translate_pydo_error(exc) from exc

    logger.info("Firewall %r deleted (id=%s)", name, existing["id"])
    return True
