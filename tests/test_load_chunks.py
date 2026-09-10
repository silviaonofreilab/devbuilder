"""Text parsing, chunking, and chunk save/load round-trips."""

from devbuilder.pipelines.preprocessing import (
    Chapter,
    chunk_chapters,
    load_chapters,
    load_chunks,
    parse_text,
    save_chapters,
    save_chunks,
)

SAMPLE_TEXT = """CHAPTER I.
Down the Rabbit-Hole

Alice was beginning to get very tired of sitting by her sister on the bank. She had peeped into the book her sister was reading, but it had no pictures or conversations in it. So she was considering, in her own mind, whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies.

CHAPTER II.
The Pool of Tears

Curiouser and curiouser! cried Alice. She was so much surprised that for the moment she quite forgot how to speak good English. Now I'm opening out like the largest telescope that ever was! Goodbye, feet!
"""


def test_parse_text_preserves_chapter_boundaries():
    chapters = parse_text(SAMPLE_TEXT)

    assert [(chapter.number, chapter.title) for chapter in chapters] == [
        (1, "Down the Rabbit-Hole"),
        (2, "The Pool of Tears"),
    ]
    assert all(chapter.content for chapter in chapters)
    assert "Curiouser" not in chapters[0].content
    assert "Curiouser" in chapters[1].content


def test_chunking_keeps_over_budget_sentence_intact():
    long_sentence = ("A" * 1500) + "."
    chapter = Chapter(number=1, title="Test", content=long_sentence)
    chunks = chunk_chapters([chapter], chunk_size=512, chunk_overlap_sentences=0)

    assert len(chunks) == 1
    assert chunks[0].text == long_sentence


def test_chunk_chapters_preserves_overlap_and_metadata():
    content = (
        "Alice followed the White Rabbit. "
        "She entered a very deep tunnel. "
        "Then she began to fall slowly."
    )
    chapters = [
        Chapter(number=1, title="First", content=content),
        Chapter(number=2, title="Second", content=content),
    ]

    chunks = chunk_chapters(
        chapters,
        chunk_size=70,
        chunk_overlap_sentences=1,
    )

    expected_texts = [
        "Alice followed the White Rabbit. She entered a very deep tunnel.",
        "She entered a very deep tunnel. Then she began to fall slowly.",
    ]
    assert [chunk.text for chunk in chunks] == expected_texts * 2
    assert [(chunk.chapter, chunk.title, chunk.chunk_index) for chunk in chunks] == [
        (1, "First", 0),
        (1, "First", 1),
        (2, "Second", 0),
        (2, "Second", 1),
    ]


def test_empty_inputs():
    assert parse_text("") == []
    assert chunk_chapters([], chunk_size=200, chunk_overlap_sentences=1) == []


def test_roundtrip_save_load(tmp_path):
    chapters = parse_text(SAMPLE_TEXT)
    chunks = chunk_chapters(chapters, chunk_size=200, chunk_overlap_sentences=1)

    save_chapters(chapters, processed_dir=tmp_path)
    save_chunks(chunks, processed_dir=tmp_path)

    assert load_chapters(processed_dir=tmp_path) == chapters
    assert load_chunks(processed_dir=tmp_path) == chunks
