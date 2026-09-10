from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class PodConfig:
    """
    Runtime configuration for a vLLM-on-RunPod deployment.
    """

    model_id: str
    gpu_type: str
    gpu_count: int
    cloud_type: str
    container_disk_gb: int
    volume_gb: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    vllm_port: int
    name: str = "devbuilder-vllm"
    image: str = "vllm/vllm-openai:v0.28.0"
    volume_mount: str = "/workspace"
    env: tuple[tuple[str, str], ...] = (
        ("HF_HOME", "/workspace/hf"),
        ("HF_TOKEN", "{{ RUNPOD_SECRET_hf_token }}"),
        ("VLLM_API_KEY", "{{ RUNPOD_SECRET_vllm_api_key }}"),
    )

    def __post_init__(self) -> None:
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be at least 1")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if not 1 <= self.vllm_port <= 65535:
            raise ValueError("vllm_port must be between 1 and 65535")

    @property
    def ports(self) -> str:
        """
        RunPod port spec for the vLLM HTTP server.
        """
        return f"{self.vllm_port}/http"

    @property
    def docker_args(self) -> str:
        """
        vLLM server CLI flags assembled from config fields.
        """
        return (
            f"--model {self.model_id} "
            f"--max-model-len {self.max_model_len} "
            f"--gpu-memory-utilization {self.gpu_memory_utilization} "
            f"--max-num-seqs {self.max_num_seqs} "
            f"--port {self.vllm_port} "
            f"--tensor-parallel-size {self.gpu_count}"
        )
