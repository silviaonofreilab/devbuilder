"""
Test policy.

Two kinds of test need something beyond the code:

- ``integration``: downloads models and runs ONNX export. Slow, network.
- ``artifact("encoder"|"reranker")``: needs that model already exported
  under artifacts/models. Skipped with a hint when it is not.

Nothing in the suite reads ``.env``, the deployment state, or a remote
service. The pod is exercised by `devbuilder-deploy pod up`, whose smoke
tests cover what a live test would.
"""

from __future__ import annotations

import pytest

import devbuilder.config as cfg
from devbuilder.paths import paths

ARTIFACT_SECTIONS = {"encoder": "encoder", "reranker": "reranker"}


def _artifact_present(settings: dict, kind: str) -> bool:
    model_name = settings[ARTIFACT_SECTIONS[kind]]["model_name"]
    return (paths.models / model_name / "manifest.json").is_file()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    settings = cfg.load_config(paths.configs / "settings.yaml")
    artifact_skips = {
        kind: pytest.mark.skip(reason=f"{kind} artifact missing; run `make export-{kind}` first")
        for kind in ARTIFACT_SECTIONS
        if not _artifact_present(settings, kind)
    }

    for item in items:
        for marker in item.iter_markers("artifact"):
            for kind in marker.args:
                if kind not in ARTIFACT_SECTIONS:
                    raise pytest.UsageError(f"Unknown artifact kind {kind!r} on {item.nodeid}")
                if kind in artifact_skips:
                    item.add_marker(artifact_skips[kind])
