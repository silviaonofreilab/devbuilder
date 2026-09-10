from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class DropletConfig:
    """
    Runtime configuration for a DigitalOcean droplet hosting DevBuilder.
    """

    ssh_key_fingerprint: str
    name: str = "devbuilder-vps"
    region: str = "sfo3"
    size: str = "s-2vcpu-4gb"
    image: str = "ubuntu-24-04-x64"
    user_data: str | None = None
    tags: tuple[str, ...] = ("devbuilder",)
