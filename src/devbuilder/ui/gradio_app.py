"""Lightweight Gradio UI for the DevBuilder RAG API."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8080")
TIMEOUT = 60.0


def _build_client() -> httpx.Client:
    """Build an authenticated client; fail fast if the token is missing."""
    token = os.environ.get("API_TOKEN")
    if not token:
        raise SystemExit("API_TOKEN is not set")
    return httpx.Client(
        base_url=API_BASE_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )


def _format_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "_No sources returned._"
    parts = []
    for i, s in enumerate(sources, 1):
        chapter = s.get("chapter") or "?"
        score = s.get("score", 0.0)
        text = s.get("text", "").strip()[:300]
        parts.append(f"**{i}. Chapter {chapter}** — score `{score:.3f}`\n\n> {text}")
    return "\n\n---\n\n".join(parts)


def ask(client: httpx.Client, query: str) -> tuple[str, str]:
    if not query.strip():
        return "Enter a question.", ""
    try:
        resp = client.post("/ask", json={"query": query})
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return f"API error: {e}", ""
    except ValueError:
        return "API error: malformed response", ""

    answer = data.get("answer", "(no answer)")
    sources = data.get("sources", [])
    return answer, _format_sources(sources)


def main() -> None:
    client = _build_client()

    def on_ask(query: str) -> tuple[str, str]:
        return ask(client, query)

    with gr.Blocks(title="DevBuilder") as demo:
        gr.Markdown("# DevBuilder RAG")
        gr.Markdown(
            "A deployment testing ground for a RAG system. "
            "*Alice in Wonderland* serves as a compact fixture corpus."
        )
        query = gr.Textbox(label="Question", lines=2, placeholder="Who is the Hatter?")
        submit = gr.Button("Ask")
        answer = gr.Textbox(label="Answer", lines=6, interactive=False)
        with gr.Accordion("Retrieved sources", open=False):
            sources = gr.Markdown()
        submit.click(on_ask, inputs=query, outputs=[answer, sources])
        query.submit(on_ask, inputs=query, outputs=[answer, sources])

    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
