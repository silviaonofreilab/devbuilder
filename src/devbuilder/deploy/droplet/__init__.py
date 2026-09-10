"""
CPU droplet on DigitalOcean: the box that runs the VPS stack (Qdrant, API,
UI, cloudflared).

``config`` describes the droplet, ``manager`` and ``firewall`` talk to the
DigitalOcean API, ``lifecycle`` sequences up/status/destroy and records
state, ``sync`` pushes config and data to the box over SSH.
"""

from __future__ import annotations

from devbuilder.deploy.droplet.config import DropletConfig
from devbuilder.deploy.droplet.firewall import (
    DropletConfigError,
    delete_firewall,
    ensure_firewall,
    resolve_ssh_cidr,
)
from devbuilder.deploy.droplet.manager import (
    DropletAuthError,
    DropletError,
    DropletNotReadyError,
    DropletQuotaError,
    create_droplet,
    destroy_droplet,
    find_droplet_by_name,
    get_client,
    get_status,
    smoke_test_ssh,
    wait_until_ready,
)

from . import lifecycle, sync  # noqa: E402  (after the names they re-export)

__all__ = [
    "DropletAuthError",
    "DropletConfig",
    "DropletConfigError",
    "DropletError",
    "DropletNotReadyError",
    "DropletQuotaError",
    "create_droplet",
    "delete_firewall",
    "destroy_droplet",
    "ensure_firewall",
    "find_droplet_by_name",
    "get_client",
    "get_status",
    "lifecycle",
    "resolve_ssh_cidr",
    "smoke_test_ssh",
    "sync",
    "wait_until_ready",
]
