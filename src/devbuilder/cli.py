from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import devbuilder.config as cfg
from devbuilder.decoder import VLLMDecoder
from devbuilder.encoder import OrtEncoder
from devbuilder.paths import paths
from devbuilder.pipelines.preprocessing import (
    chunk_chapters,
    load_chapters,
    load_chunks,
    parse_text,
    save_chapters,
    save_chunks,
)
from devbuilder.rag import RAG
from devbuilder.reranker import OrtReranker
from devbuilder.vectorstore import QdrantVectorStore


def _make_encoder(settings: dict) -> OrtEncoder:
    enc_cfg = settings["encoder"]

    return OrtEncoder(
        models_dir=paths.models,
        model_name=enc_cfg["model_name"],
        batch_size=enc_cfg["batch_size"],
    )


def _make_reranker(settings: dict) -> OrtReranker:
    rr_cfg = settings["reranker"]

    return OrtReranker(
        models_dir=paths.models,
        model_name=rr_cfg["model_name"],
        batch_size=rr_cfg["batch_size"],
    )


def _make_store(encoder: OrtEncoder, settings: dict) -> QdrantVectorStore:
    vs_cfg = settings["vectorstore"]

    return QdrantVectorStore(
        embeddings=encoder,
        store_name=vs_cfg["name"],
        host=vs_cfg.get("host", "localhost"),
        port=vs_cfg.get("port", 6333),
    )


def _make_decoder(settings: dict) -> VLLMDecoder:

    return VLLMDecoder.from_settings(
        settings["decoder"],
        system_prompt=settings["prompts"]["system"],
    )


def _make_rag(store: QdrantVectorStore, settings: dict, top_k: int) -> RAG:
    prompt_cfg = settings["prompts"]
    rr_cfg = settings["reranker"]

    return RAG(
        vectorstore=store,
        template=prompt_cfg["rag_template"],
        top_k=top_k,
        reranker=_make_reranker(settings),
        fetch_k=rr_cfg["fetch_k"],
    )


def _print_artifact(artifact) -> None:
    """
    Report the ONNX files produced by an export.
    """
    print(f"\nExported: {artifact.output_dir}")
    print(f"Base ONNX: {artifact.base_onnx_path}")
    if artifact.optimized_onnx_path:
        print(f"Optimized: {artifact.optimized_onnx_path}")
    if artifact.quantized_onnx_path:
        print(f"Quantized: {artifact.quantized_onnx_path}")


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Parsers

    parser = argparse.ArgumentParser(prog="devbuilder")
    parser.add_argument(
        "--config",
        default=str(paths.configs / "settings.yaml"),
        help="Path to YAML config file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("parse", help="Parse raw text into chapters.")
    subparsers.add_parser("chunk", help="Chunk chapters for RAG.")
    subparsers.add_parser("pipeline", help="Run full text processing pipeline.")
    index_parser = subparsers.add_parser("index", help="Compute embeddings and build Qdrant index.")
    index_parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Drop and rebuild the collection if it already exists.",
    )

    query_parser = subparsers.add_parser("query", help="Query the vector store.")
    query_parser.add_argument(
        "query", type=str, nargs="?", help="Query text OR test key (e.g. mad_hatter)"
    )
    query_parser.add_argument("-k", "--top-k", type=int, help="Override default top-k")
    query_parser.add_argument(
        "--examples", action="store_true", help="Run configured example queries"
    )
    query_parser.add_argument(
        "--with-decoder",
        action="store_true",
        help="Generate an answer via vLLM after retrieval.",
    )

    export_parser = subparsers.add_parser(
        "export-encoder", help="Export encoder model to ONNX via Optimum."
    )
    export_parser.add_argument("--model-id", type=str, help="Override config model_id")
    export_parser.add_argument("--model-name", type=str, help="Override config model name")
    export_parser.add_argument("--optimize", type=str, help="O1/O2/O3/O4")
    export_parser.add_argument(
        "--quantize", type=str, help="int8 (dynamic INT8 weight quantization) or omit"
    )
    export_parser.add_argument("--validate", action="store_true", help="Run validation")

    rerank_parser = subparsers.add_parser(
        "export-reranker", help="Export reranker model to ONNX via Optimum."
    )
    rerank_parser.add_argument("--model-id", type=str, help="Override config model_id")
    rerank_parser.add_argument("--model-name", type=str, help="Override config model name")
    rerank_parser.add_argument("--optimize", type=str, help="O1/O2/O3/O4")
    rerank_parser.add_argument(
        "--quantize", type=str, help="int8 (dynamic INT8 weight quantization) or omit"
    )
    rerank_parser.add_argument("--validate", action="store_true", help="Run validation")

    args = parser.parse_args()

    settings = cfg.load_config(args.config)
    data_cfg = settings["data"]
    vs_cfg = settings["vectorstore"]
    enc_cfg = settings["encoder"]

    # Handlers

    if args.command == "parse":
        paths.ensure_dirs()
        text = (paths.raw / data_cfg["raw_file"]).read_text(encoding="utf-8")
        chapters = parse_text(text)
        save_chapters(chapters, paths.processed)
        print(f"Parsed {len(chapters)} chapters -> data/processed/chapters.json")

    elif args.command == "chunk":
        chapters = load_chapters(paths.processed)
        chunks = chunk_chapters(
            chapters,
            chunk_size=data_cfg["chunk_size"],
            chunk_overlap_sentences=data_cfg["chunk_overlap_sentences"],
        )
        save_chunks(chunks, paths.processed)
        print(f"Created {len(chunks)} chunks -> data/processed/chunks.json")

    elif args.command == "pipeline":
        paths.ensure_dirs()
        text = (paths.raw / data_cfg["raw_file"]).read_text(encoding="utf-8")
        chapters = parse_text(text)
        save_chapters(chapters, paths.processed)
        print(f"Parsed {len(chapters)} chapters")

        chunks = chunk_chapters(
            chapters,
            chunk_size=data_cfg["chunk_size"],
            chunk_overlap_sentences=data_cfg["chunk_overlap_sentences"],
        )
        save_chunks(chunks, paths.processed)
        print(f"Created {len(chunks)} chunks")

    elif args.command == "index":
        chunks = load_chunks(paths.processed)
        print(f"Loading {len(chunks)} chunks...")

        encoder = _make_encoder(settings=settings)
        print(f"Encoder ready ({encoder.model_name})")

        store = _make_store(encoder, settings=settings)
        print(f"Building Qdrant collection ({store.store_name})...")
        store.build_index(chunks, force_recreate=args.force_recreate)
        print(f"Indexed {len(chunks)} chunks -> collection '{store.store_name}'")

    elif args.command == "query":
        encoder = _make_encoder(settings=settings)
        store = _make_store(encoder, settings=settings)

        examples = settings["examples"]["queries"]
        top_k = args.top_k or vs_cfg["top_k_default"]

        decoder_client = None
        rag = None
        if args.with_decoder:
            decoder_client = _make_decoder(settings)
            rag = _make_rag(
                store,
                settings,
                top_k=top_k,
            )

        run_log = {
            "timestamp": datetime.now(UTC).isoformat(),
            "encoder_model": encoder.model_name,
            "vectorstore": store.store_name,
            "examples": {},
        }

        def run_one(q: str) -> dict:
            results = store.retrieve(q, top_k=top_k)
            results = sorted(results, key=lambda x: x[1], reverse=True)

            rows = []
            print(f"\n=== {q} ===\n")
            for i, (payload, score) in enumerate(results, 1):
                ch = payload.get("chapter")
                idx = payload.get("chunk_index")
                text = payload.get("text", "")[:200]

                print(f"[{i}] Chapter {ch}, Chunk {idx} (score: {score:.4f})")
                print(f"    {text}...\n")

                rows.append(
                    {
                        "rank": i,
                        "chapter": ch,
                        "chunk_index": idx,
                        "score": float(score),
                        "text": text,
                    }
                )

            reranked_rows = None
            answer = None
            contexts = None
            if rag is not None:
                # Retrieve twice on purpose: the raw hits above, then the reranked
                # set, so the effect of the reranker is visible side by side.
                scored = rag.retrieve_with_scores(q)
                contexts = [payload.get("text", "") for payload, _ in scored]
                print("--- Reranked ---\n")
                reranked_rows = []

                for i, (payload, score) in enumerate(scored, 1):
                    ch, idx = payload.get("chapter"), payload.get("chunk_index")
                    print(f"[{i}] Chapter {ch}, Chunk {idx} (score: {score:.4f})")
                    reranked_rows.append(
                        {"rank": i, "chapter": ch, "chunk_index": idx, "score": float(score)}
                    )

            if decoder_client is not None:
                prompt = rag.enhance_user_prompt(q, contexts)
                answer = decoder_client.generate(prompt)
                print(f"\n--- Answer ---\n{answer}\n")

            return {"results": rows, "reranked": reranked_rows, "answer": answer}

        if args.examples:
            if not examples:
                raise SystemExit("No examples configured under examples.queries")
            for key, q in examples.items():
                print(f"\n--- example: {key} ---")
                run_log["examples"][key] = {"query": q, **run_one(q)}
        else:
            if args.query in examples:
                q = examples[args.query]
            elif args.query:
                q = args.query
            else:
                raise SystemExit("Provide a query, a key from examples.queries, or --examples")
            run_log["examples"]["adhoc"] = {"query": q, **run_one(q)}

        runs_dir = paths.runs
        runs_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        out = runs_dir / f"{ts}_{store.store_name}.json"

        with open(out, "w") as f:
            json.dump(run_log, f, indent=2)

        print(f"\nSaved run log → {out}")

    elif args.command == "export-encoder":
        from devbuilder.export import OnnxBuilder

        model_id = args.model_id or enc_cfg["model_id"]
        model_name = args.model_name or enc_cfg["model_name"]
        optimize = args.optimize if args.optimize is not None else enc_cfg.get("onnx_optimize")
        quantize = args.quantize if args.quantize is not None else enc_cfg.get("onnx_quantize")
        pooling = enc_cfg.get("pooling", "mean")

        builder = OnnxBuilder(models_dir=paths.models)
        artifact = builder.export_encoder(
            model_id=model_id,
            revision=enc_cfg.get("revision"),
            model_name=model_name,
            embedding_dim=enc_cfg["embedding_dim"],
            normalize=enc_cfg["normalize"],
            max_length=enc_cfg["max_length"],
            pooling=pooling,
            optimize=optimize,
            quantize=quantize,
            validate=args.validate,
        )
        _print_artifact(artifact)

    elif args.command == "export-reranker":
        from devbuilder.export import OnnxBuilder

        rr_cfg = settings["reranker"]
        model_id = args.model_id or rr_cfg["model_id"]
        model_name = args.model_name or rr_cfg["model_name"]
        optimize = args.optimize if args.optimize is not None else rr_cfg.get("onnx_optimize")
        quantize = args.quantize if args.quantize is not None else rr_cfg.get("onnx_quantize")

        builder = OnnxBuilder(models_dir=paths.models)
        artifact = builder.export_reranker(
            model_id=model_id,
            revision=rr_cfg.get("revision"),
            model_name=model_name,
            max_length=rr_cfg["max_length"],
            optimize=optimize,
            quantize=quantize,
            validate=args.validate,
        )
        _print_artifact(artifact)


if __name__ == "__main__":
    main()
