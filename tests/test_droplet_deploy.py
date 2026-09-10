"""Droplet deployment orchestration, with the DigitalOcean primitives mocked."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from devbuilder.deploy.droplet import DropletError, lifecycle

IP = "203.0.113.10"


@pytest.fixture
def deploy_mocks():
    """Patch the DigitalOcean primitives and deployment-state writes used by deploy."""
    targets = (
        "create_droplet",
        "delete_firewall",
        "destroy_droplet",
        "ensure_firewall",
        "find_droplet_by_name",
        "get_status",
        "resolve_ssh_cidr",
        "smoke_test_ssh",
        "wait_until_ready",
        "update_state",
        "clear_state",
    )
    with ExitStack() as stack:
        mocks = {name: stack.enter_context(patch.object(lifecycle, name)) for name in targets}
        mocks["create_droplet"].return_value = (123, None)
        mocks["wait_until_ready"].return_value = IP
        mocks["resolve_ssh_cidr"].return_value = "198.51.100.1/32"
        mocks["ensure_firewall"].return_value = "fw-1"
        yield SimpleNamespace(**mocks)


@pytest.fixture
def config() -> SimpleNamespace:
    return SimpleNamespace(name="demo")


def test_up_records_each_resource_before_the_next_step(deploy_mocks, config) -> None:
    order: list[str] = []
    deploy_mocks.update_state.side_effect = lambda **kw: order.append(f"state:{','.join(kw)}")
    deploy_mocks.ensure_firewall.side_effect = lambda **kw: order.append("firewall") or "fw-1"
    deploy_mocks.smoke_test_ssh.side_effect = lambda ip: order.append("ssh")

    lifecycle.up(config)

    assert order == [
        "state:droplet_id",
        "firewall",
        "state:firewall_id",
        "state:droplet_ip",
        "ssh",
    ]
    deploy_mocks.update_state.assert_any_call(droplet_ip=IP)
    deploy_mocks.update_state.assert_any_call(firewall_id="fw-1")
    deploy_mocks.ensure_firewall.assert_called_once_with(
        name="demo-fw", droplet_id=123, ssh_source_cidr="198.51.100.1/32"
    )


def test_unresolvable_ssh_cidr_creates_nothing(deploy_mocks, config) -> None:
    # The CIDR is needed for the firewall; failing to resolve it must not cost money.
    deploy_mocks.resolve_ssh_cidr.side_effect = DropletError("no public IP")
    with pytest.raises(DropletError):
        lifecycle.up(config)
    deploy_mocks.create_droplet.assert_not_called()


def test_firewall_is_attached_before_waiting_for_the_box(deploy_mocks, config) -> None:
    order: list[str] = []
    deploy_mocks.ensure_firewall.side_effect = lambda **kw: order.append("firewall") or "fw-1"
    deploy_mocks.wait_until_ready.side_effect = lambda droplet_id: order.append("wait") or IP

    lifecycle.up(config)

    assert order.index("firewall") < order.index("wait")


def test_ssh_failure_preserves_droplet_for_retry(deploy_mocks, config) -> None:
    deploy_mocks.smoke_test_ssh.side_effect = DropletError("SSH failed")
    with pytest.raises(DropletError):
        lifecycle.up(config)
    deploy_mocks.destroy_droplet.assert_not_called()
    deploy_mocks.clear_state.assert_not_called()


@pytest.mark.parametrize(
    "existing", [pytest.param(None, id="absent"), pytest.param({"id": 123}, id="present")]
)
def test_destroy_removes_firewall_and_clears_host(deploy_mocks, config, existing) -> None:
    deploy_mocks.find_droplet_by_name.return_value = existing

    lifecycle.destroy(config)

    deploy_mocks.delete_firewall.assert_called_once_with("demo-fw")
    deploy_mocks.clear_state.assert_called_once_with("droplet_id", "droplet_ip", "firewall_id")
    assert deploy_mocks.destroy_droplet.called is (existing is not None)
