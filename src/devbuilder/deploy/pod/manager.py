from __future__ import annotations

import logging
import os
import time

import httpx
import runpod
from runpod.error import AuthenticationError

from devbuilder.deploy.pod.config import PodConfig

logger = logging.getLogger(__name__)


# Exceptions


class PodError(Exception):
    """
    Base for all RunPod operational errors.
    """


class PodAuthError(PodError):
    """
    Invalid or missing RUNPOD_API_KEY.
    """


class PodQuotaError(PodError):
    """
    No GPUs of the requested type available.
    """


class PodNotReadyError(PodError):
    """
    Pod did not become ready or failed a deployment smoke test.
    """


class VLLMAuthError(PodError):
    """
    Missing or rejected VLLM_API_KEY.
    """


def _translate_runpod_error(exc: Exception) -> PodError:
    """
    Map RunPod SDK exceptions to our typed exceptions.
    """
    msg = str(exc).lower()
    if isinstance(exc, AuthenticationError) or "unauthorized" in msg:
        return PodAuthError("RUNPOD_API_KEY rejected by RunPod API")
    if "no available" in msg or "out of capacity" in msg or "quota" in msg:
        return PodQuotaError(str(exc))
    return PodError(str(exc))


# Credentials


def _require_api_key() -> str:
    key = os.getenv("RUNPOD_API_KEY")
    if not key:
        raise PodAuthError("RUNPOD_API_KEY not set in environment")
    return key


def configure_runpod() -> None:
    runpod.api_key = _require_api_key()


# Helpers


def find_pod_by_name(name: str) -> dict | None:
    """
    Return the pod dict matching `name`, or None if absent.
    """
    configure_runpod()
    try:
        pods = runpod.get_pods()
    except Exception as exc:
        raise _translate_runpod_error(exc) from exc
    for pod in pods:
        if pod.get("name") == name:
            return pod
    return None


def endpoint_for(pod_id: str, port: int) -> str:
    """
    Public RunPod proxy URL for a pod and port. Deterministic in the pod id,
    so it survives stop/resume and never needs to be stored separately from
    the id (it is stored anyway, for humans reading the state file).
    """
    return f"https://{pod_id}-{port}.proxy.runpod.net"


# Lifecycle


def get_status(pod_id: str) -> dict | None:
    """
    Return pod info dict, or None if the pod no longer exists.
    """
    try:
        return runpod.get_pod(pod_id)
    except Exception as exc:
        raise _translate_runpod_error(exc) from exc


def resume_pod(pod_id: str, gpu_count: int) -> None:
    """
    Start a stopped pod on its existing volume.
    """
    try:
        runpod.resume_pod(pod_id, gpu_count=gpu_count)
    except Exception as exc:
        raise _translate_runpod_error(exc) from exc
    logger.info("Pod %s resumed", pod_id)


def create_pod(config: PodConfig) -> tuple[str, str]:
    """
    Ensure a pod with this name is running: reuse a running one, resume a
    stopped one, or create a new one. Returns (pod_id, endpoint).
    """

    existing = find_pod_by_name(config.name)
    if existing is not None:
        pod_id = existing["id"]
        status = get_status(pod_id) or {}
        if status.get("desiredStatus") == "EXITED":
            logger.info("Pod %r exists but is stopped (id=%s); resuming", config.name, pod_id)
            resume_pod(pod_id, config.gpu_count)
        else:
            logger.info("Pod %r already running (id=%s); reusing", config.name, pod_id)
        return pod_id, endpoint_for(pod_id, config.vllm_port)

    logger.info("Creating pod %r (gpu=%s, model=%s)", config.name, config.gpu_type, config.model_id)
    try:
        pod = runpod.create_pod(
            name=config.name,
            image_name=config.image,
            gpu_type_id=config.gpu_type,
            gpu_count=config.gpu_count,
            cloud_type=config.cloud_type,
            container_disk_in_gb=config.container_disk_gb,
            volume_in_gb=config.volume_gb,
            volume_mount_path=config.volume_mount,
            ports=config.ports,
            docker_args=config.docker_args,
            env=dict(config.env),
        )
    except Exception as exc:
        raise _translate_runpod_error(exc) from exc

    endpoint = endpoint_for(pod["id"], config.vllm_port)
    logger.info("Pod created (id=%s, endpoint=%s)", pod["id"], endpoint)
    return pod["id"], endpoint


def _probe_health(endpoint: str) -> int:
    """
    Perform a single /health probe.
    """
    response = httpx.get(f"{endpoint}/health", timeout=5)
    return response.status_code


def wait_until_ready(endpoint: str, timeout: int = 600, interval: int = 2) -> None:
    """
    Poll /health until 200 or timeout (seconds). Idempotent.
    """
    logger.info("Waiting for vLLM at %s (timeout=%ds)", endpoint, timeout)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _probe_health(endpoint) == 200:
                logger.info("vLLM ready at %s", endpoint)
                return
        except httpx.HTTPError as exc:
            logger.debug("Health probe failed: %s", exc)
        time.sleep(interval)

    raise PodNotReadyError(f"{endpoint}/health not 200 within {timeout}s")


def smoke_test_health(endpoint: str) -> None:
    """
    Assert /health returns 200.
    """
    try:
        status = _probe_health(endpoint)
    except httpx.HTTPError as exc:
        raise PodNotReadyError(f"Health request failed: {exc}") from exc

    if status != 200:
        raise PodNotReadyError(f"/health returned {status}")
    logger.info("Health smoke test passed")


def smoke_test_completion(endpoint: str, model_id: str) -> None:
    """
    Assert /v1/chat/completions returns a non-empty authenticated response.
    """
    token = os.getenv("VLLM_API_KEY")
    if not token:
        raise VLLMAuthError("VLLM_API_KEY not set in environment")

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 8,
    }

    try:
        response = httpx.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise PodNotReadyError(f"Completion request failed: {exc}") from exc

    if response.status_code == 401:
        raise VLLMAuthError("VLLM_API_KEY rejected by vLLM")

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise PodNotReadyError(f"Completion request returned {response.status_code}") from exc

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise PodNotReadyError("Completion returned a malformed response") from exc

    if not isinstance(content, str) or not content.strip():
        raise PodNotReadyError("Completion returned empty content")

    logger.info("Completion smoke test passed (response=%r)", content[:40])


def smoke_test_auth_rejected(endpoint: str, model_id: str) -> None:
    """
    Assert an unauthenticated /v1 request is rejected with 401.
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    }

    try:
        response = httpx.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise PodNotReadyError(f"Auth smoke-test request failed: {exc}") from exc

    if response.status_code != 401:
        raise PodNotReadyError(
            f"Unauthenticated /v1 request returned {response.status_code}, "
            "expected 401; auth gate is open"
        )

    logger.info("Auth smoke test passed (unauthenticated request rejected)")


def stop_pod(pod_id: str) -> None:
    """
    Stop the pod, releasing the GPU. The volume and pod id survive, so the
    endpoint is unchanged after ``resume_pod``. No-op if stopped or absent.
    """

    pod = get_status(pod_id)
    if pod is None:
        logger.info("stop_pod: pod %s not found; nothing to do", pod_id)
        return
    if pod.get("desiredStatus") == "EXITED":
        logger.info("stop_pod: pod %s already stopped", pod_id)
        return
    try:
        runpod.stop_pod(pod_id)
    except Exception as exc:
        raise _translate_runpod_error(exc) from exc
    logger.info("Pod %s stopped", pod_id)


def terminate_pod(pod_id: str) -> None:
    """
    Permanently delete pod and its volume. No-op if absent.
    """
    if get_status(pod_id) is None:
        logger.info("terminate_pod: pod %s not found; nothing to do", pod_id)
        return

    try:
        runpod.terminate_pod(pod_id)
    except Exception as exc:
        raise _translate_runpod_error(exc) from exc

    logger.info("Pod %s terminated", pod_id)
