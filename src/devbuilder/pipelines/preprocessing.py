from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Alice in Wonderland has 12 chapters; explicit map keeps the fixture self-contained.
_ALICE_ROMAN_MAP: dict[str, int] = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}

# Drop chunks shorter than this — filters out section breaks, stray fragments.
MIN_CHUNK_CHARS = 40


@dataclass
class Chapter:
    number: int | None = None
    title: str | None = None
    content: str | None = None


@dataclass
class Chunk:
    text: str
    chapter: int | None = None
    title: str | None = None
    chunk_index: int = 0


def parse_text(text: str) -> list[Chapter]:
    """
    Parse Alice in Wonderland from Project Gutenberg into chapters.
    """
    logger.debug("Parsing raw text (%d chars)", len(text))

    # Remove header (everything before first chapter)
    text = re.split(r"CHAPTER I\.\n", text, maxsplit=1)[-1]
    text = "CHAPTER I.\n" + text

    # Remove decorative asterisks
    text = re.sub(r"\n\s*\*[\s\*]+\n", "\n\n", text)
    # Remove emphasis markers
    text = text.replace("_", "")

    # Remove everything after the end of the last chapter
    text = re.split(r"THE END", text, maxsplit=1)[0]

    # Split into chapters
    chapter_pattern = r"CHAPTER ([IVXLC]+)\.\n([^\n]+)\n"
    parts = re.split(chapter_pattern, text)

    chapters = []
    for i in range(1, len(parts), 3):
        if i + 2 < len(parts):
            roman = parts[i].strip()
            title = parts[i + 1].strip()
            content = parts[i + 2].strip()

            num = _ALICE_ROMAN_MAP.get(roman, 0)
            if num == 0:
                logger.warning("Unrecognized chapter numeral %r — defaulting to 0", roman)

            chapters.append(Chapter(number=num, title=title, content=content))

    logger.info("Parsed %d chapters", len(chapters))
    return chapters


def save_chapters(chapters: list[Chapter], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "chapters.json", "w") as f:
        json.dump([asdict(ch) for ch in chapters], f, indent=2)
    logger.info("Saved %d chapters to %s", len(chapters), processed_dir / "chapters.json")


def load_chapters(processed_dir: Path) -> list[Chapter]:
    with open(processed_dir / "chapters.json") as f:
        return [Chapter(**ch) for ch in json.load(f)]


def chunk_by_sentences(text: str, chunk_size: int, chunk_overlap_sentences: int) -> list[str]:
    """
    Chunk text by sentences, respecting target chunk size.
    """
    # Collapse the source's hard line breaks and runs of spaces; chunks are
    # embedded and displayed as prose, not as typeset lines.
    text = re.sub(r"\s+", " ", text).strip()

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_size + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep last N sentences for overlap
            current_chunk = (
                current_chunk[-chunk_overlap_sentences:] if chunk_overlap_sentences else []
            )
            current_size = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_size += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_chapters(
    chapters: list[Chapter], chunk_size: int, chunk_overlap_sentences: int
) -> list[Chunk]:
    logger.debug(
        "Chunking %d chapters (chunk_size=%d, overlap_sentences=%d)",
        len(chapters),
        chunk_size,
        chunk_overlap_sentences,
    )

    chunks: list[Chunk] = []
    for chapter in chapters:
        chapter_chunks = chunk_by_sentences(
            chapter.content or "", chunk_size, chunk_overlap_sentences
        )

        filtered = [t.strip() for t in chapter_chunks if len(t.strip()) >= MIN_CHUNK_CHARS]
        for idx, cleaned in enumerate(filtered):
            chunks.append(
                Chunk(
                    chapter=chapter.number or 0,
                    title=chapter.title,
                    chunk_index=idx,
                    text=cleaned,
                )
            )
    logger.info("Produced %d chunks", len(chunks))
    return chunks


def save_chunks(chunks: list[Chunk], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([asdict(ch) for ch in chunks], f, indent=2)
    logger.info("Saved %d chunks to %s", len(chunks), processed_dir / "chunks.json")


def load_chunks(processed_dir: Path) -> list[Chunk]:
    with open(processed_dir / "chunks.json", encoding="utf-8") as f:
        return [Chunk(**ch) for ch in json.load(f)]
