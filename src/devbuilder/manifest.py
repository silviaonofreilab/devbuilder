"""Artifact manifests.

Records the contract between the export pipeline and any consumer
(indexer, api). Written by the export job, read at startup by indexer
and api to verify the loaded artifact matches what the runtime expects.

Covers both encoder and reranker artifacts; each manifest lives alongside
its ONNX file in the artifact directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Self, TypedDict, TypeVar

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
TOKENIZER_SUBDIR = "tokenizer"
MANIFEST_VERSION = 3

# Main Classes


@dataclass(frozen=True)
class ExportedWith:
    """
    Versions of the toolchain used to produce the ONNX artifact.
    """

    optimum: str
    onnxruntime: str
    transformers: str
    python: str


@dataclass(frozen=True)
class ArtifactManifest:
    """
    Common contract for an exported ONNX artifact.

    Consumers must reject manifest versions they do not understand.
    """

    manifest_version: int
    model_id: str
    model_revision: str | None
    model_name: str
    max_length: int
    optimize: str | None
    quantize: str | None
    onnx_opset: int
    onnx_ir_version: int
    artifact_filename: str
    artifact_sha256: str
    tokenizer_sha256: str
    exported_with: ExportedWith
    exported_at: str

    def to_json(self) -> str:
        """Serialize the manifest to stable, human-readable JSON."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Construct a manifest from a parsed dictionary."""
        data = dict(data)
        exported_with = ExportedWith(**data.pop("exported_with"))
        return cls(exported_with=exported_with, **data)


@dataclass(frozen=True)
class EncoderManifest(ArtifactManifest):
    """Contract for an exported encoder artifact."""

    embedding_dim: int
    normalize: bool
    pooling: str


@dataclass(frozen=True)
class RerankerManifest(ArtifactManifest):
    """Contract for an exported reranker artifact."""


M = TypeVar("M", bound=ArtifactManifest)


# Errors


class ManifestError(Exception):
    """Base for manifest-related problems."""


class ManifestMissingError(ManifestError):
    """No manifest.json found in the expected location."""


class ManifestReadError(ManifestError):
    """Raised when a manifest file cannot be read."""


class ManifestSchemaError(ManifestError):
    """Manifest exists but doesn't match the expected schema."""


class IncompatibleArtifactError(ManifestError):
    """Raised when the manifest version or artifact contract is unsupported."""


class CorruptArtifactError(ManifestError):
    """Raised when an artifact's checksum differs from its manifest."""


# Handlers


def write_manifest(manifest: ArtifactManifest, output_dir: Path) -> Path:
    """
    Write the manifest json file into the export directory.

    Args:
        manifest: Fully-constructed manifest.
        output_dir: Artifact root (the directory that also contains ``onnx/``).

    Returns:
        Path to the written manifest file.
    """
    path = output_dir / MANIFEST_FILENAME
    path.write_text(manifest.to_json())
    logger.info("Wrote manifest to %s", path)
    return path


def read_manifest(output_dir: Path, manifest_cls: type[M]) -> M:
    """
    Read the manifest json file from an artifact directory.

    Args:
        output_dir: Artifact root (the directory that also contains ``onnx/``).
        manifest_cls: Which manifest type to parse into
            (``EncoderManifest`` or ``RerankerManifest``).

    Returns:
        Parsed manifest of type ``manifest_cls``.

    Raises:
        ManifestMissingError: If the file doesn't exist.
        ManifestReadError: If the file cannot be read.
        ManifestSchemaError: If the file exists but doesn't parse or is
            missing required fields.
    """
    path = output_dir / MANIFEST_FILENAME

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ManifestMissingError(f"Manifest not found: {path}") from exc
    except OSError as exc:
        raise ManifestReadError(f"Could not read manifest: {path}") from exc

    try:
        data = json.loads(raw)
        return manifest_cls.from_dict(data)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        raise ManifestSchemaError(f"Manifest at {path} is invalid: {e}") from e


class _CommonManifestFields(TypedDict):
    manifest_version: int
    model_id: str
    model_revision: str | None
    model_name: str
    max_length: int
    optimize: str | None
    quantize: str | None
    onnx_opset: int
    onnx_ir_version: int
    artifact_filename: str
    artifact_sha256: str
    tokenizer_sha256: str
    exported_with: ExportedWith
    exported_at: str


def _common_fields(
    *,
    model_id: str,
    model_revision: str | None,
    model_name: str,
    max_length: int,
    optimize: str | None,
    quantize: str | None,
    active_onnx_path: Path,
    output_dir: Path,
) -> _CommonManifestFields:
    """Build fields shared by encoder and reranker manifests."""
    opset, ir_version = _onnx_versions(active_onnx_path)

    return {
        "manifest_version": MANIFEST_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_name": model_name,
        "max_length": max_length,
        "optimize": optimize,
        "quantize": quantize,
        "onnx_opset": opset,
        "onnx_ir_version": ir_version,
        "artifact_filename": str(active_onnx_path.relative_to(output_dir)),
        "artifact_sha256": _sha256(active_onnx_path),
        "tokenizer_sha256": tokenizer_sha256(output_dir / TOKENIZER_SUBDIR),
        "exported_with": _current_toolchain(),
        "exported_at": datetime.now(UTC).isoformat(),
    }


def build_encoder_manifest(
    *,
    model_id: str,
    model_revision: str | None,
    model_name: str,
    embedding_dim: int,
    normalize: bool,
    max_length: int,
    pooling: str,
    optimize: str | None,
    quantize: str | None,
    active_onnx_path: Path,
    output_dir: Path,
) -> EncoderManifest:
    """Construct an encoder manifest from artifact and runtime state."""
    return EncoderManifest(
        **_common_fields(
            model_id=model_id,
            model_revision=model_revision,
            model_name=model_name,
            max_length=max_length,
            optimize=optimize,
            quantize=quantize,
            active_onnx_path=active_onnx_path,
            output_dir=output_dir,
        ),
        embedding_dim=embedding_dim,
        normalize=normalize,
        pooling=pooling,
    )


def build_reranker_manifest(
    *,
    model_id: str,
    model_revision: str | None,
    model_name: str,
    max_length: int,
    optimize: str | None,
    quantize: str | None,
    active_onnx_path: Path,
    output_dir: Path,
) -> RerankerManifest:
    """Construct a reranker manifest from artifact and runtime state."""
    return RerankerManifest(
        **_common_fields(
            model_id=model_id,
            model_revision=model_revision,
            model_name=model_name,
            max_length=max_length,
            optimize=optimize,
            quantize=quantize,
            active_onnx_path=active_onnx_path,
            output_dir=output_dir,
        )
    )


# Embedding identity


def embedding_fingerprint(manifest: EncoderManifest) -> str:
    """
    SHA-256 over everything that determines an encoder's output vectors:
    the ONNX graph, the tokenizer files, pooling, normalization, and the
    input length. Two encoders with the same fingerprint produce the same
    embeddings for the same text; an index is stamped with it and refuses
    any encoder whose fingerprint differs. Timestamps and toolchain versions
    are deliberately excluded.
    """
    h = hashlib.sha256()
    for part in (
        manifest.artifact_sha256,
        manifest.tokenizer_sha256,
        manifest.pooling,
        str(manifest.normalize),
        str(manifest.max_length),
    ):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


# Helpers


def _sha256(path: Path) -> str:
    """
    Stream-hash a file in 1 MiB chunks (avoids loading large ONNX into memory).
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_sha256(tokenizer_dir: Path) -> str:
    """
    One hash over every tokenizer file (name and content, sorted), so a
    changed vocabulary or normalizer is caught like a changed graph.
    """
    h = hashlib.sha256()
    for path in sorted(p for p in tokenizer_dir.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(tokenizer_dir)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _onnx_versions(active_onnx_path: Path) -> tuple[int, int]:
    """
    Extract (opset, ir_version) from an ONNX file without resolving tensor data.

    Export-time only: ``onnx`` ships with the ``export`` extra, not the runtime
    images, so it is imported here rather than at module scope.
    """
    import onnx

    proto = onnx.load(str(active_onnx_path), load_external_data=False)
    opset = max(o.version for o in proto.opset_import) if proto.opset_import else 0
    return opset, proto.ir_version


def _current_toolchain() -> ExportedWith:
    """
    Snapshot the installed export toolchain versions.
    """
    return ExportedWith(
        optimum=_pkg_version("optimum"),
        onnxruntime=_pkg_version("onnxruntime"),
        transformers=_pkg_version("transformers"),
        python=sys.version.split()[0],
    )


def _pkg_version(name: str) -> str:
    """
    Look up an installed package version; return ``"unknown"`` if not found.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


# Verification


def verify_manifest(
    manifest: ArtifactManifest,
    *,
    model_dir: Path,
) -> None:
    """
    Verify that a manifest and its ONNX artifact are compatible and intact.

    Args:
        manifest: Parsed encoder or reranker manifest.
        model_dir: Artifact directory containing the referenced ONNX file.

    Raises:
        CorruptArtifactError: If the ONNX file or the tokenizer files don't
            match the manifest's checksums.
    """
    if manifest.manifest_version != MANIFEST_VERSION:
        raise IncompatibleArtifactError(
            f"Unsupported manifest version {manifest.manifest_version}; "
            f"expected {MANIFEST_VERSION}."
        )

    artifact_path = model_dir / manifest.artifact_filename
    actual_sha = _sha256(artifact_path)
    if actual_sha != manifest.artifact_sha256:
        raise CorruptArtifactError(
            f"sha mismatch for {artifact_path}: "
            f"manifest={manifest.artifact_sha256[:12]}, actual={actual_sha[:12]}"
        )

    tokenizer_dir = model_dir / TOKENIZER_SUBDIR
    actual_tok = tokenizer_sha256(tokenizer_dir)
    if actual_tok != manifest.tokenizer_sha256:
        raise CorruptArtifactError(
            f"tokenizer sha mismatch for {tokenizer_dir}: "
            f"manifest={manifest.tokenizer_sha256[:12]}, actual={actual_tok[:12]}"
        )

    runtime_ort = _pkg_version("onnxruntime")
    if runtime_ort != manifest.exported_with.onnxruntime:
        logger.warning(
            "ORT version drift: runtime=%s, exported=%s",
            runtime_ort,
            manifest.exported_with.onnxruntime,
        )
