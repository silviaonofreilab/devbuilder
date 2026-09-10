"""
Pod lifecycle (RunPod): sequence the provider operations and record the
result in the deployment state. Provider calls live in ``manager``.
"""

from __future__ import annotations

import logging

from devbuilder.deploy.pod.config import PodConfig
from devbuilder.deploy.pod.manager import (
    create_pod,
    find_pod_by_name,
    get_status,
    smoke_test_auth_rejected,
    smoke_test_completion,
    smoke_test_health,
    stop_pod,
    terminate_pod,
    wait_until_ready,
)
from devbuilder.deploy.state import clear_state, update_state

logger = logging.getLogger(__name__)

STATE_FIELDS = ("pod_id", "vllm_endpoint")


def up(config: PodConfig) -> None:
    """
    Create, resume, or reuse the pod; wait for readiness; run smoke tests.
    State is written as soon as the pod exists, so a failing check still
    leaves a reachable pod recorded for diagnosis.
    """
    pod_id, endpoint = create_pod(config)
    update_state(pod_id=pod_id, vllm_endpoint=endpoint)
    wait_until_ready(endpoint)

    smoke_test_health(endpoint)
    smoke_test_completion(endpoint, config.model_id)
    smoke_test_auth_rejected(endpoint, config.model_id)
    logger.info("Pod %s is up and serving at %s", pod_id, endpoint)


def status(config: PodConfig) -> None:
    """
    Report pod state as RunPod sees it.
    """
    pod = find_pod_by_name(config.name)
    if pod is None:
        logger.info("No pod named %r found", config.name)
        return

    current = get_status(pod["id"]) or {}
    logger.info(
        "Pod %r: id=%s, desired=%s, runtime=%s",
        config.name,
        pod["id"],
        current.get("desiredStatus"),
        current.get("runtime"),
    )


def stop(config: PodConfig) -> None:
    """
    Stop the pod, releasing the GPU while keeping the volume. The endpoint is
    a function of the pod id, so state is left as is; ``up`` resumes it.
    """
    pod = find_pod_by_name(config.name)
    if pod is None:
        logger.info("No pod named %r found; nothing to stop", config.name)
        return
    stop_pod(pod["id"])


def terminate(config: PodConfig) -> None:
    """
    Permanently delete the pod and its volume, and forget it.
    """
    pod = find_pod_by_name(config.name)
    if pod is not None:
        terminate_pod(pod["id"])
    else:
        logger.info("No pod named %r found; nothing to terminate", config.name)

    clear_state(*STATE_FIELDS)
