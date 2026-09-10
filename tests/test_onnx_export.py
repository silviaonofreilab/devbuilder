"""ONNX export across optimize/quantize topologies, and loading the result."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from onnxruntime import SessionOptions

import devbuilder.config as cfg
from devbuilder.encoder import OrtEncoder
from devbuilder.export import OnnxBuilder
from devbuilder.manifest import (
    CorruptArtifactError,
    EncoderManifest,
    ManifestSchemaError,
    RerankerManifest,
    read_manifest,
)
from devbuilder.paths import paths
from devbuilder.reranker import OrtReranker

pytestmark = pytest.mark.integration


# (optimize, quantize, expected_active_filename)
TOPOLOGIES = [
    (None, None, "model.onnx"),
    ("O3", None, "model_O3.onnx"),
    (None, "int8", "model_qint8.onnx"),
    ("O3", "int8", "model_O3_qint8.onnx"),
]


@pytest.fixture(scope="module")
def settings():
    return cfg.load_config(paths.configs / "settings.yaml")


# Encoder


@pytest.fixture(scope="module", params=TOPOLOGIES, ids=["base", "opt", "quant", "chain"])
def exported_encoder(request, tmp_path_factory, settings):
    enc_cfg = settings["encoder"]

    optimize, quantize, active = request.param
    tmp_path = tmp_path_factory.mktemp("onnx")
    builder = OnnxBuilder(models_dir=tmp_path)
    artifact = builder.export_encoder(
        model_id=enc_cfg["model_id"],
        model_name="miniLM_test",
        embedding_dim=enc_cfg["embedding_dim"],
        normalize=enc_cfg["normalize"],
        max_length=enc_cfg["max_length"],
        pooling=enc_cfg.get("pooling", "mean"),
        optimize=optimize,
        quantize=quantize,
        validate=True,
    )
    return artifact, tmp_path, active


def test_export_smoke(exported_encoder):
    artifact, _, _ = exported_encoder
    assert artifact.output_dir.exists()
    assert artifact.tokenizer_dir.exists()
    assert artifact.base_onnx_path.exists()


def test_manifest_active_file(exported_encoder, settings):
    artifact, _, active = exported_encoder
    manifest = read_manifest(artifact.output_dir, EncoderManifest)

    assert manifest.artifact_filename == f"onnx/{active}"
    assert manifest.model_id == settings["encoder"]["model_id"]
    assert (artifact.output_dir / manifest.artifact_filename).is_file()


def test_encoder_loads_and_encodes(exported_encoder, settings):
    artifact, tmp_path, _ = exported_encoder
    manifest = read_manifest(artifact.output_dir, EncoderManifest)
    expected_dim = settings["encoder"]["embedding_dim"]

    encoder = OrtEncoder(
        models_dir=tmp_path,
        model_name="miniLM_test",
        batch_size=2,
    )
    output = encoder.embed_texts(["Who is the Hatter?"])

    assert manifest.embedding_dim == expected_dim
    assert output.shape == (1, manifest.embedding_dim)
    assert np.isfinite(output).all()


def test_encoder_applies_explicit_session_options(exported_encoder):
    _, models_dir, _ = exported_encoder
    options = SessionOptions()
    options.intra_op_num_threads = 1

    encoder = OrtEncoder(
        models_dir=models_dir,
        model_name="miniLM_test",
        batch_size=2,
        provider="CPUExecutionProvider",
        session_options=options,
    )

    effective_options = encoder._session.get_session_options()
    assert effective_options.intra_op_num_threads == 1


# Reranker


# Base export, and the topology the deployment actually uses (settings.yaml).
RERANKER_TOPOLOGIES = [
    (None, None, "model.onnx"),
    ("O3", "int8", "model_O3_qint8.onnx"),
]


@pytest.fixture(scope="module", params=RERANKER_TOPOLOGIES, ids=["base", "deployed"])
def exported_reranker(request, tmp_path_factory, settings):
    rr_cfg = settings["reranker"]
    optimize, quantize, active = request.param

    tmp_path = tmp_path_factory.mktemp("onnx_rerank")
    builder = OnnxBuilder(models_dir=tmp_path)
    artifact = builder.export_reranker(
        model_id=rr_cfg["model_id"],
        model_name="reranker_test",
        max_length=rr_cfg["max_length"],
        optimize=optimize,
        quantize=quantize,
        validate=True,
    )
    return artifact, tmp_path, active


def test_reranker_manifest(exported_reranker, settings):
    artifact, _, active = exported_reranker
    manifest = read_manifest(artifact.output_dir, RerankerManifest)
    assert manifest.artifact_filename == f"onnx/{active}"
    assert manifest.model_id == settings["reranker"]["model_id"]


def test_manifest_rejects_wrong_dataclass(exported_reranker):
    artifact, _, _ = exported_reranker

    with pytest.raises(ManifestSchemaError):
        read_manifest(artifact.output_dir, EncoderManifest)


def test_reranker_scores_and_ranks(exported_reranker):
    _, tmp_path, _ = exported_reranker
    reranker = OrtReranker(models_dir=tmp_path, model_name="reranker_test", batch_size=4)
    query = "Who is the Hatter?"
    passages = [
        "The Hatter is a character at the mad tea party.",
        "The Cheshire Cat can disappear at will.",
    ]
    scores = reranker.score(query, passages)
    assert scores.shape == (len(passages),)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]  # relevant passage ranks higher
    assert scores[1] < 0.0  # raw logit, not a sigmoid probability


def test_reranker_empty_passages(exported_reranker):
    _, tmp_path, _ = exported_reranker
    reranker = OrtReranker(models_dir=tmp_path, model_name="reranker_test", batch_size=4)
    assert reranker.score("q", []).shape == (0,)


# Integrity of the whole artifact directory


def test_manifest_records_revision_and_tokenizer_hash(exported_reranker):
    artifact, _, _ = exported_reranker
    manifest = read_manifest(artifact.output_dir, RerankerManifest)
    # Nothing is pinned in settings; the export resolves and records the Hub commit anyway.
    assert manifest.model_revision is not None and len(manifest.model_revision) == 40
    assert len(manifest.tokenizer_sha256) == 64


def test_tampered_tokenizer_is_refused(exported_reranker, tmp_path_factory):
    # Copy the artifact so the module-scoped fixture stays intact.
    artifact, _, _ = exported_reranker
    root = tmp_path_factory.mktemp("tampered")
    shutil.copytree(artifact.output_dir, root / "reranker_test")
    (root / "reranker_test" / "tokenizer" / "tokenizer_config.json").write_text("{}")

    with pytest.raises(CorruptArtifactError, match="tokenizer"):
        OrtReranker(models_dir=root, model_name="reranker_test", batch_size=1)


def test_failed_export_keeps_previous_artifact(exported_reranker, settings, monkeypatch):
    artifact, models_dir, _ = exported_reranker
    before = read_manifest(artifact.output_dir, RerankerManifest)
    rr_cfg = settings["reranker"]

    def explode(*args, **kwargs):
        raise RuntimeError("optimizer crashed")

    monkeypatch.setattr(OnnxBuilder, "_postprocess", staticmethod(explode))
    with pytest.raises(RuntimeError):
        OnnxBuilder(models_dir=models_dir).export_reranker(
            model_id=rr_cfg["model_id"],
            model_name="reranker_test",
            max_length=rr_cfg["max_length"],
            optimize="O3",
        )

    assert read_manifest(artifact.output_dir, RerankerManifest) == before
    assert not any(p.name.endswith(".partial") for p in models_dir.iterdir())


def test_failed_promotion_restores_previous_artifact(exported_reranker, settings, monkeypatch):
    artifact, models_dir, _ = exported_reranker
    before = read_manifest(artifact.output_dir, RerankerManifest)
    rr_cfg = settings["reranker"]

    real_rename = Path.rename

    def rename_fails_into_final(self: Path, target):
        if self.name.endswith(".partial"):  # only the promotion of the new export
            raise OSError("disk full")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename_fails_into_final)
    with pytest.raises(OSError):
        OnnxBuilder(models_dir=models_dir).export_reranker(
            model_id=rr_cfg["model_id"],
            model_name="reranker_test",
            max_length=rr_cfg["max_length"],
        )

    assert read_manifest(artifact.output_dir, RerankerManifest) == before
    leftovers = {p.name for p in models_dir.iterdir()} - {"reranker_test"}
    assert not leftovers, leftovers
