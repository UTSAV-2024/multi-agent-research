# ==========================================
# CHUNK REPOSITORY TESTS
# ==========================================
#
# Tests for save_chunks() with focus on
# start/end field persistence.
#
# ==========================================

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, AsyncMock


class TestSaveChunks:
    """Tests for save_chunks()."""

    # ---------------------------------------------------------------
    # HELPER
    # ---------------------------------------------------------------

    def _run(self, coro):
        """Run an async test."""
        return asyncio.run(coro)

    # ---------------------------------------------------------------
    # TESTS: start AND end PRESENT
    # ---------------------------------------------------------------

    @patch(
        "app.repositories.chunk_repository.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_persists_start_and_end_when_present(
        self,
        mock_collection,
    ):
        """When start/end exist in chunk, they are passed to insert_many."""
        from app.repositories.chunk_repository import save_chunks

        chunks = [
            {
                "chunk_id": 1,
                "content": "First chunk content.",
                "title": "Source 1",
                "url": "https://example.com/1",
                "start": 0,
                "end": 100,
            },
            {
                "chunk_id": 2,
                "content": "Second chunk content.",
                "title": "Source 1",
                "url": "https://example.com/1",
                "start": 100,
                "end": 200,
            },
        ]

        self._run(save_chunks(
            report_id="test_report_001",
            source_url="https://example.com/1",
            chunks=chunks,
        ))

        # Verify insert_many was called once
        mock_collection.insert_many.assert_awaited_once()

        # Extract the documents passed to insert_many
        call_args = mock_collection.insert_many.call_args[0][0]

        assert len(call_args) == 2

        # Chunk 1: start=0, end=100
        assert call_args[0]["start"] == 0
        assert call_args[0]["end"] == 100
        assert call_args[0]["chunk_id"] == 1

        # Chunk 2: start=100, end=200
        assert call_args[1]["start"] == 100
        assert call_args[1]["end"] == 200
        assert call_args[1]["chunk_id"] == 2

    @patch(
        "app.repositories.chunk_repository.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_persists_zero_start_end(
        self,
        mock_collection,
    ):
        """Explicit start=0 and end=0 are passed through correctly."""
        from app.repositories.chunk_repository import save_chunks

        chunks = [
            {
                "chunk_id": 1,
                "content": "Zero-length chunk.",
                "title": "Source",
                "url": "https://example.com",
                "start": 0,
                "end": 0,
            },
        ]

        self._run(save_chunks(
            report_id="test_report_002",
            source_url="https://example.com",
            chunks=chunks,
        ))

        call_args = mock_collection.insert_many.call_args[0][0]

        assert call_args[0]["start"] == 0
        assert call_args[0]["end"] == 0

    @patch(
        "app.repositories.chunk_repository.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_persists_nonzero_start_end(
        self,
        mock_collection,
    ):
        """Non-zero start/end values are passed through correctly."""
        from app.repositories.chunk_repository import save_chunks

        chunks = [
            {
                "chunk_id": 5,
                "content": "Mid-document chunk.",
                "title": "Source",
                "url": "https://example.com",
                "start": 350,
                "end": 720,
            },
        ]

        self._run(save_chunks(
            report_id="test_report_003",
            source_url="https://example.com",
            chunks=chunks,
        ))

        call_args = mock_collection.insert_many.call_args[0][0]

        assert call_args[0]["start"] == 350
        assert call_args[0]["end"] == 720

    # ---------------------------------------------------------------
    # TESTS: start AND end ABSENT
    # ---------------------------------------------------------------

    @patch(
        "app.repositories.chunk_repository.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_succeeds_when_start_end_absent(
        self,
        mock_collection,
    ):
        """When start/end are missing, fallback to 0, no crash."""
        from app.repositories.chunk_repository import save_chunks

        chunks = [
            {
                "chunk_id": 1,
                "content": "Chunk without start/end.",
                "title": "Source",
                "url": "https://example.com",
                # deliberately no "start", no "end"
            },
        ]

        self._run(save_chunks(
            report_id="test_report_004",
            source_url="https://example.com",
            chunks=chunks,
        ))

        mock_collection.insert_many.assert_awaited_once()

        call_args = mock_collection.insert_many.call_args[0][0]

        assert len(call_args) == 1
        assert call_args[0]["start"] == 0  # fallback
        assert call_args[0]["end"] == 0    # fallback
        assert call_args[0]["chunk_id"] == 1
        assert call_args[0]["content"] == "Chunk without start/end."

    @patch(
        "app.repositories.chunk_repository.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_succeeds_with_partial_metadata(
        self,
        mock_collection,
    ):
        """Chunks with minimal metadata (only chunk_id + content) still persist."""
        from app.repositories.chunk_repository import save_chunks

        chunks = [
            {
                "chunk_id": 7,
                "content": "Minimal chunk.",
                # no title, no url, no start, no end
            },
        ]

        self._run(save_chunks(
            report_id="test_report_005",
            source_url="https://example.com",
            chunks=chunks,
        ))

        mock_collection.insert_many.assert_awaited_once()

        call_args = mock_collection.insert_many.call_args[0][0]

        assert len(call_args) == 1
        assert call_args[0]["start"] == 0  # fallback
        assert call_args[0]["end"] == 0    # fallback
        assert call_args[0]["title"] == ""  # fallback
        assert call_args[0]["url"] == ""    # fallback
        assert call_args[0]["report_id"] == "test_report_005"
        assert call_args[0]["source_url"] == "https://example.com"

    @patch(
        "app.repositories.chunk_repository.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_empty_chunks_list(
        self,
        mock_collection,
    ):
        """Empty chunks list does not call insert_many."""
        from app.repositories.chunk_repository import save_chunks

        result = self._run(save_chunks(
            report_id="test_report_006",
            source_url="https://example.com",
            chunks=[],
        ))

        assert result is None
        mock_collection.insert_many.assert_not_called()

    # ---------------------------------------------------------------
    # TESTS: MIXED (some chunks have start/end, some don't)
    # ---------------------------------------------------------------

    @patch(
        "app.repositories.chunk_repository.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_mixed_chunks(
        self,
        mock_collection,
    ):
        """Mixed chunks (some with start/end, some without) all persist correctly."""
        from app.repositories.chunk_repository import save_chunks

        chunks = [
            {
                "chunk_id": 1,
                "content": "Has start/end.",
                "title": "A",
                "url": "https://a.com",
                "start": 0,
                "end": 50,
            },
            {
                "chunk_id": 2,
                "content": "Missing start/end.",
                "title": "B",
                "url": "https://b.com",
                # no start, no end
            },
            {
                "chunk_id": 3,
                "content": "Has start only.",
                "title": "C",
                "url": "https://c.com",
                "start": 100,
                # no end
            },
        ]

        self._run(save_chunks(
            report_id="test_report_007",
            source_url="https://example.com",
            chunks=chunks,
        ))

        call_args = mock_collection.insert_many.call_args[0][0]

        assert len(call_args) == 3

        # Chunk 1: start=0, end=50
        assert call_args[0]["start"] == 0
        assert call_args[0]["end"] == 50

        # Chunk 2: missing both -> fallback to 0
        assert call_args[1]["start"] == 0
        assert call_args[1]["end"] == 0

        # Chunk 3: has start=100, missing end -> end defaults to 0
        assert call_args[2]["start"] == 100
        assert call_args[2]["end"] == 0

    # ---------------------------------------------------------------
    # TESTS: RETURN VALUE
    # ---------------------------------------------------------------

    @patch(
        "app.repositories.chunk_repository.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_returns_chunk_count(
        self,
        mock_collection,
    ):
        """save_chunks returns the number of stored chunks."""
        from app.repositories.chunk_repository import save_chunks

        chunks = [
            {"chunk_id": 1, "content": "One."},
            {"chunk_id": 2, "content": "Two."},
            {"chunk_id": 3, "content": "Three."},
        ]

        result = self._run(save_chunks(
            report_id="test_report_008",
            source_url="https://example.com",
            chunks=chunks,
        ))

        assert result == 3
