"""
Export HF transformer models (encoder and cross-encoder) to ONNX via Optimum,
with optional optimization and quantization.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from huggingface_hub import model_info
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
    ORTOptimizer,
)
from optimum.onnxruntime.configuration import AutoOptimizationConfig
from transformers import AutoTokenizer

from devbuilder.manifest import (
    TOKENIZER_SUBDIR,
    build_encoder_manifest,
    build_reranker_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)

OptimizeLevel = Literal["O1", "O2", "O3", "O4"]
# The only quantization offered is ONNX Runtime's dynamic INT8 weight
# quantization (`quantize_dynamic`, QInt8). It is not tuned per CPU
# architecture; the resulting graph runs on any ORT CPU provider.
QuantizeTarget = Literal["int8"]

ONNX_SUBDIR = "onnx"
BASE_ONNX_NAME = "model.onnx"
STAGING_SUFFIX = ".partial"
BACKUP_SUFFIX = ".previous"


@dataclass(frozen=True)
class OnnxArtifact:
    """
    Paths produced by an ONNX export.

    Attributes:
        model_id: HuggingFace model ID that was exported.
        output_dir: Root directory containing the saved model and tokenizer.
        onnx_dir: Subdirectory holding all *.onnx files (output_dir / "onnx").
        base_onnx_path: Path to the unoptimized base export (onnx/model.onnx).
        optimized_onnx_path: Path to the optimized model, if optimize was set.
        quantized_onnx_path: Path to the quantized model, if quantize was set.
        tokenizer_dir: Directory containing the tokenizer files.
    """

    model_id: str
    output_dir: Path
    onnx_dir: Path
    base_onnx_path: Path
    tokenizer_dir: Path
    optimized_onnx_path: Path | None = None
    quantized_onnx_path: Path | None = None


class OnnxBuilder:
    """
    Export HF encoder and reranker models to ONNX using Optimum.

    Produces a task-appropriate base ONNX graph via Optimum, then optionally
    optimizes it with ORTOptimizer and dynamically quantizes it with ONNX
    Runtime. All model variants are written to the onnx/ subdirectory.
    """

    def __init__(self, models_dir: Path) -> None:
        """
        Initialize the builder.

        Args:
            models_dir: Root directory under which exported models are written.
                Each export creates a models_dir / model_name subdirectory.
        """
        self.models_dir = models_dir
        logger.debug("OnnxBuilder initialized (models_dir=%s)", models_dir)

    @staticmethod
    def _postprocess(
        onnx_dir: Path,
        base_onnx_path: Path,
        optimize: OptimizeLevel | None,
        quantize: QuantizeTarget | None,
    ) -> tuple[Path | None, Path | None]:
        """
        Run the optional optimize + quantize stages on the base graph.

        Returns:
            (optimized_onnx_path, quantized_onnx_path); each None if skipped.
        """
        optimized_onnx_path: Path | None = None
        if optimize:
            optimized_onnx_path = OnnxBuilder._optimize(onnx_dir, optimize)
            _assert_exists(optimized_onnx_path, f"optimize={optimize}")

        quantized_onnx_path: Path | None = None
        if quantize:
            source = optimized_onnx_path or base_onnx_path
            quantized_onnx_path = OnnxBuilder._quantize(source, onnx_dir, quantize)
            _assert_exists(quantized_onnx_path, f"quantize={quantize}")

        return optimized_onnx_path, quantized_onnx_path

    def export_encoder(
        self,
        model_id: str,
        model_name: str,
        *,
        revision: str | None = None,
        embedding_dim: int,
        normalize: bool,
        max_length: int,
        pooling: str = "mean",
        optimize: OptimizeLevel | None = None,
        quantize: QuantizeTarget | None = None,
        validate: bool = False,
        provider: str = "CPUExecutionProvider",
        trust_remote_code: bool = False,
    ) -> OnnxArtifact:
        """
        Performs the base export, optionally optimizes and quantizes the graph.

        Args:
            model_id: HuggingFace model ID (e.g. "sentence-transformers/all-MiniLM-L6-v2").
            revision: Hub revision (commit hash, tag, or branch) to export; None
                means the Hub default branch. A commit hash makes the export
                reproducible; it is recorded in the manifest either way.
            model_name: Output directory name under models_dir.
            embedding_dim: Vector dimensionality the encoder produces.
            normalize: Whether the runtime should L2-normalize pooled output.
            max_length: Encoder token limit recorded for runtime tokenization and
                used by export validation.
            pooling: Pooling strategy recorded in the manifest ("mean"/"cls"/"max").
            optimize: Optimization level passed to ORTOptimizer.
            quantize: "int8" for dynamic INT8 weight quantization, or None.
            validate: If True, load the produced graph with ORT and run one forward pass.
            provider: ORT execution provider for the base export.
            trust_remote_code: Allow remote code when loading the model/tokenizer.

        Returns:
            An :class: OnnxArtifact with explicit paths to every file produced.

        Raises:
            FileNotFoundError: If an expected output file is missing after a stage.
        """
        try:
            revision = _resolve_revision(model_id, revision)
            artifact = self._export(
                model_id=model_id,
                revision=revision,
                model_name=model_name,
                model_kind="encoder",
                ort_model_cls=ORTModelForFeatureExtraction,
                validation_output_attr="last_hidden_state",
                validate_as_pair=False,
                max_length=max_length,
                optimize=optimize,
                quantize=quantize,
                validate=validate,
                provider=provider,
                trust_remote_code=trust_remote_code,
            )

            active_path = (
                artifact.quantized_onnx_path
                or artifact.optimized_onnx_path
                or artifact.base_onnx_path
            )
            manifest = build_encoder_manifest(
                model_id=model_id,
                model_revision=revision,
                model_name=model_name,
                embedding_dim=embedding_dim,
                normalize=normalize,
                max_length=max_length,
                pooling=pooling,
                optimize=optimize,
                quantize=quantize,
                active_onnx_path=active_path,
                output_dir=artifact.output_dir,
            )
            write_manifest(manifest, artifact.output_dir)
            artifact = self._commit(artifact, model_name)
        except BaseException:
            self._discard_staging(model_name)
            raise

        logger.info("Exported encoder %s to %s", model_id, artifact.output_dir)
        return artifact

    def export_reranker(
        self,
        model_id: str,
        model_name: str,
        *,
        revision: str | None = None,
        max_length: int,
        optimize: OptimizeLevel | None = None,
        quantize: QuantizeTarget | None = None,
        validate: bool = False,
        provider: str = "CPUExecutionProvider",
        trust_remote_code: bool = False,
    ) -> OnnxArtifact:
        """
        Export a sequence-classification cross-encoder for reranking.

        Args:
            model_id: HuggingFace reranker model ID.
            revision: Hub revision to export (see ``export_encoder``).
            model_name: Output directory name under models_dir.
            max_length: Reranker pair-token limit recorded for runtime tokenization
                and used by export validation.
            optimize: Optimization level passed to ORTOptimizer.
            quantize: "int8" for dynamic INT8 weight quantization, or None.
            validate: If True, load the produced graph and score a query-passage pair.
            provider: ORT execution provider for the base export.
            trust_remote_code: Allow remote code when loading the model/tokenizer.

        Returns:
            An :class:`OnnxArtifact` with explicit paths to every file produced.

        Raises:
            FileNotFoundError: If an expected output file is missing after a stage.
        """
        try:
            revision = _resolve_revision(model_id, revision)
            artifact = self._export(
                model_id=model_id,
                revision=revision,
                model_name=model_name,
                model_kind="reranker",
                ort_model_cls=ORTModelForSequenceClassification,
                validation_output_attr="logits",
                validate_as_pair=True,
                max_length=max_length,
                optimize=optimize,
                quantize=quantize,
                validate=validate,
                provider=provider,
                trust_remote_code=trust_remote_code,
            )

            active_path = (
                artifact.quantized_onnx_path
                or artifact.optimized_onnx_path
                or artifact.base_onnx_path
            )
            manifest = build_reranker_manifest(
                model_id=model_id,
                model_revision=revision,
                model_name=model_name,
                max_length=max_length,
                optimize=optimize,
                quantize=quantize,
                active_onnx_path=active_path,
                output_dir=artifact.output_dir,
            )
            write_manifest(manifest, artifact.output_dir)
            artifact = self._commit(artifact, model_name)
        except BaseException:
            self._discard_staging(model_name)
            raise
        logger.info("Exported reranker %s to %s", model_id, artifact.output_dir)
        return artifact

    def _export(
        self,
        model_id: str,
        model_name: str,
        *,
        revision: str | None = None,
        model_kind: str,
        ort_model_cls: type,
        validation_output_attr: str,
        validate_as_pair: bool,
        max_length: int,
        optimize: OptimizeLevel | None,
        quantize: QuantizeTarget | None,
        validate: bool,
        provider: str,
        trust_remote_code: bool,
    ) -> OnnxArtifact:
        """Run the shared base-export, tokenizer, post-processing, and validation pipeline."""
        out_dir = self._prepare_output_dir(model_name)
        onnx_dir = out_dir / ONNX_SUBDIR
        onnx_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Exporting %s %s to ONNX at %s (provider=%s)",
            model_kind,
            model_id,
            out_dir,
            provider,
        )
        ort_model = ort_model_cls.from_pretrained(
            model_id,
            revision=revision,
            export=True,
            provider=provider,
            trust_remote_code=trust_remote_code,
        )
        ort_model.save_pretrained(onnx_dir)

        base_onnx_path = onnx_dir / BASE_ONNX_NAME
        _assert_exists(base_onnx_path, "base export")
        logger.debug("Base export produced %s", base_onnx_path)

        tokenizer_dir = self._save_tokenizer(model_id, revision, out_dir, trust_remote_code)
        optimized_onnx_path, quantized_onnx_path = self._postprocess(
            onnx_dir, base_onnx_path, optimize, quantize
        )

        artifact = OnnxArtifact(
            model_id=model_id,
            output_dir=out_dir,
            onnx_dir=onnx_dir,
            base_onnx_path=base_onnx_path,
            tokenizer_dir=tokenizer_dir,
            optimized_onnx_path=optimized_onnx_path,
            quantized_onnx_path=quantized_onnx_path,
        )

        if validate:
            self._validate(
                artifact=artifact,
                ort_model_cls=ort_model_cls,
                output_attr=validation_output_attr,
                provider=provider,
                trust_remote_code=trust_remote_code,
                max_length=max_length,
                as_pair=validate_as_pair,
            )

        return artifact

    def _prepare_output_dir(self, model_name: str) -> Path:
        """
        Prepare a clean staging directory for model_name. The export is built
        here and moved into place by ``_commit`` only once everything,
        including the manifest, has been written, so a failed export never
        removes the previous artifact.
        """
        staging = self.models_dir / f"{model_name}{STAGING_SUFFIX}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        return staging

    def _discard_staging(self, model_name: str) -> None:
        """Remove a failed export's staging directory; the previous artifact is untouched."""
        staging = self.models_dir / f"{model_name}{STAGING_SUFFIX}"
        if staging.exists():
            shutil.rmtree(staging)
            logger.info("Discarded partial export %s", staging)

    def _commit(self, artifact: OnnxArtifact, model_name: str) -> OnnxArtifact:
        """
        Replace models_dir / model_name with the staged export and return the
        artifact with its paths rebased onto the final directory. The previous
        artifact is restored if the new one cannot be moved into place.
        """
        staging = artifact.output_dir
        final = self.models_dir / model_name
        backup = self.models_dir / f"{model_name}{BACKUP_SUFFIX}"

        # Promote by rename, keeping the previous artifact aside until the new
        # one is in place; if promotion fails, put the previous one back.
        if backup.exists():
            shutil.rmtree(backup)
        had_previous = final.exists()
        if had_previous:
            final.rename(backup)
        try:
            staging.rename(final)
        except BaseException:
            if had_previous:
                backup.rename(final)
            raise
        if had_previous:
            shutil.rmtree(backup)

        def rebase(path: Path | None) -> Path | None:
            return None if path is None else final / path.relative_to(staging)

        return replace(
            artifact,
            output_dir=final,
            onnx_dir=rebase(artifact.onnx_dir),
            base_onnx_path=rebase(artifact.base_onnx_path),
            tokenizer_dir=rebase(artifact.tokenizer_dir),
            optimized_onnx_path=rebase(artifact.optimized_onnx_path),
            quantized_onnx_path=rebase(artifact.quantized_onnx_path),
        )

    @staticmethod
    def _save_tokenizer(
        model_id: str, revision: str | None, out_dir: Path, trust_remote_code: bool
    ) -> Path:
        """
        Save the tokenizer to out_dir / tokenizer and return that path.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, use_fast=True, trust_remote_code=trust_remote_code
        )
        tok_dir = out_dir / TOKENIZER_SUBDIR
        tokenizer.save_pretrained(tok_dir)
        logger.debug("Saved tokenizer to %s", tok_dir)
        return tok_dir

    @staticmethod
    def _optimize(onnx_dir: Path, optimize: OptimizeLevel) -> Path:
        """
        Optimize the base graph in place, producing model_<level>.onnx.
        """
        logger.info("Optimizing ONNX model (level=%s)", optimize)
        optimizer = ORTOptimizer.from_pretrained(str(onnx_dir))
        opt_cfg = getattr(AutoOptimizationConfig, optimize)()
        optimizer.optimize(
            optimization_config=opt_cfg,
            save_dir=str(onnx_dir),
            file_suffix=optimize,
        )
        return onnx_dir / f"model_{optimize}.onnx"

    @staticmethod
    def _quantize(source: Path, onnx_dir: Path, quantize: QuantizeTarget) -> Path:
        """
        Dynamically INT8-quantize source after symbolic shape pre-processing.

        Args:
            source: ONNX file to quantize (base or optimized).
            onnx_dir: Directory the quantized file is written into.
            quantize: Quantization scheme; only "int8" exists.

        Returns:
            Path to the quantized ONNX file.
        """
        logger.info("Quantizing ONNX model (%s, source=%s)", quantize, source.name)
        out_path = onnx_dir / f"{source.stem}_q{quantize}.onnx"
        preprocessed = onnx_dir / f"{source.stem}_pre.onnx"

        try:
            quant_pre_process(
                input_model=str(source),
                output_model_path=str(preprocessed),
                skip_optimization=True,
                auto_merge=True,
                guess_output_rank=True,
            )
            quantize_dynamic(
                model_input=str(preprocessed),
                model_output=str(out_path),
                weight_type=QuantType.QInt8,
            )
        except Exception:
            out_path.unlink(missing_ok=True)
            raise
        finally:
            preprocessed.unlink(missing_ok=True)
        return out_path

    @staticmethod
    def _validate(
        artifact: OnnxArtifact,
        ort_model_cls: type,
        output_attr: str,
        provider: str,
        trust_remote_code: bool,
        max_length: int,
        as_pair: bool,
    ) -> None:
        """
        Load the most specialized variant with ORT and run one smoke-test forward pass.
        """
        target = (
            artifact.quantized_onnx_path or artifact.optimized_onnx_path or artifact.base_onnx_path
        )
        logger.info("Validating exported model (%s)", target.name)
        model = ort_model_cls.from_pretrained(
            artifact.onnx_dir,
            file_name=target.name,
            provider=provider,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            artifact.tokenizer_dir, use_fast=True, trust_remote_code=trust_remote_code
        )
        if as_pair:
            inputs = tokenizer(
                "What is Alice looking for?",
                "Alice follows the White Rabbit down the rabbit hole.",
                truncation="only_second",
                max_length=max_length,
                return_tensors="np",
            )
        else:
            inputs = tokenizer(
                "hello world",
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
        outputs = model(**inputs)
        if getattr(outputs, output_attr, None) is None:
            raise RuntimeError(f"Validation failed: outputs missing {output_attr}.")
        logger.debug("Validation forward pass succeeded")


def _resolve_revision(model_id: str, revision: str | None) -> str | None:
    """
    The commit the Hub will serve for ``revision`` (or its default branch), so
    the manifest always names an exact upstream commit even when the config
    pins nothing. Returns ``revision`` unchanged if the Hub cannot be asked
    (offline export from cache): the export still works, the manifest just
    records what was requested.
    """
    try:
        sha = model_info(model_id, revision=revision).sha
    except Exception as exc:  # network, auth, or an unknown revision
        logger.warning("Could not resolve Hub revision for %s: %s", model_id, exc)
        return revision
    if sha and sha != revision:
        logger.info("Resolved %s revision %s -> %s", model_id, revision or "default", sha[:12])
    return sha or revision


def _assert_exists(path: Path, stage: str) -> None:
    """
    Raise FileNotFoundError with context if path is missing after a stage.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected ONNX artifact missing after {stage}: {path}.")
