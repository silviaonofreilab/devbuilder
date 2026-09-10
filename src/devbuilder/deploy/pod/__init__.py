"""
GPU pod on RunPod: vLLM serving the decoder.

``config`` describes the pod, ``manager`` talks to the RunPod API, and
``lifecycle`` sequences up/status/stop/terminate and records state.
"""

from __future__ import annotations

from devbuilder.deploy.pod.config import PodConfig
from devbuilder.deploy.pod.manager import (
    PodAuthError,
    PodError,
    PodNotReadyError,
    PodQuotaError,
    VLLMAuthError,
    configure_runpod,
    create_pod,
    endpoint_for,
    find_pod_by_name,
    get_status,
    resume_pod,
    smoke_test_auth_rejected,
    smoke_test_completion,
    smoke_test_health,
    stop_pod,
    terminate_pod,
    wait_until_ready,
)

from . import lifecycle  # noqa: E402  (after the names it re-exports)

__all__ = [
    "PodAuthError",
    "PodConfig",
    "PodError",
    "PodNotReadyError",
    "PodQuotaError",
    "VLLMAuthError",
    "configure_runpod",
    "create_pod",
    "endpoint_for",
    "find_pod_by_name",
    "get_status",
    "lifecycle",
    "resume_pod",
    "smoke_test_auth_rejected",
    "smoke_test_completion",
    "smoke_test_health",
    "stop_pod",
    "terminate_pod",
    "wait_until_ready",
]
