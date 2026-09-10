"""Embedding fingerprint: what changes it, what does not."""

from __future__ import annotations

from dataclasses import replace

from devbuilder.manifest import EncoderManifest, ExportedWith, embedding_fingerprint


def _manifest(**overrides) -> EncoderManifest:
    base = EncoderManifest(
        manifest_version=3,
        model_id="m",
        model_revision=None,
        model_name="m",
        max_length=256,
        optimize="O3",
        quantize="int8",
        onnx_opset=18,
        onnx_ir_version=8,
        artifact_filename="onnx/model.onnx",
        artifact_sha256="a" * 64,
        tokenizer_sha256="t" * 64,
        exported_with=ExportedWith(optimum="2", onnxruntime="1", transformers="4", python="3"),
        exported_at="2026-09-08T00:00:00+00:00",
        embedding_dim=384,
        normalize=True,
        pooling="mean",
    )
    return replace(base, **overrides)


def test_fingerprint_changes_with_each_embedding_input() -> None:
    reference = embedding_fingerprint(_manifest())
    for change in (
        {"artifact_sha256": "b" * 64},
        {"tokenizer_sha256": "u" * 64},
        {"pooling": "cls"},
        {"normalize": False},
        {"max_length": 128},
    ):
        assert embedding_fingerprint(_manifest(**change)) != reference, change


def test_fingerprint_ignores_incidental_metadata() -> None:
    reference = embedding_fingerprint(_manifest())
    for change in (
        {"exported_at": "2027-01-01T00:00:00+00:00"},
        {"exported_with": ExportedWith(optimum="9", onnxruntime="9", transformers="9", python="9")},
        {"model_revision": "abc123"},
        {"optimize": None},
    ):
        assert embedding_fingerprint(_manifest(**change)) == reference, change
