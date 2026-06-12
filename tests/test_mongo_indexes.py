# ==========================================
# MONGO INDEXES TESTS
# ==========================================
#
# Tests for MongoDB index creation at startup.
#
# ==========================================

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, AsyncMock


class TestEnsureChunksIndexes:
    """Tests for ensure_chunks_indexes()."""

    @patch(
        "app.db.collections.chunks_collection.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_creates_compound_index(
        self,
        mock_collection,
    ):
        """Verifies the compound index on (report_id, chunk_id) is created."""
        from app.db.collections.chunks_collection import (
            ensure_chunks_indexes,
        )

        asyncio.run(ensure_chunks_indexes())

        mock_collection.create_index.assert_awaited_once_with(
            [("report_id", 1), ("chunk_id", 1)],
            name="report_id_chunk_id",
        )

    @patch(
        "app.db.collections.chunks_collection.chunks_collection",
        new_callable=AsyncMock,
    )
    def test_idempotent_on_repeat(
        self,
        mock_collection,
    ):
        """Calling ensure_chunks_indexes twice does not raise."""
        from app.db.collections.chunks_collection import (
            ensure_chunks_indexes,
        )

        asyncio.run(ensure_chunks_indexes())
        asyncio.run(ensure_chunks_indexes())

        assert mock_collection.create_index.await_count == 2


class TestEnsureReportsIndexes:
    """Tests for ensure_reports_indexes()."""

    @patch(
        "app.db.collections.reports_collection.reports_collection",
        new_callable=AsyncMock,
    )
    def test_creates_both_indexes(
        self,
        mock_collection,
    ):
        """Verifies both indexes are created with correct specs."""
        from app.db.collections.reports_collection import (
            ensure_reports_indexes,
        )

        asyncio.run(ensure_reports_indexes())

        assert mock_collection.create_index.await_count == 2

        mock_collection.create_index.assert_any_call(
            [("request_id", 1)],
            name="request_id",
        )

        mock_collection.create_index.assert_any_call(
            [("created_at", -1)],
            name="created_at_desc",
        )

    @patch(
        "app.db.collections.reports_collection.reports_collection",
        new_callable=AsyncMock,
    )
    def test_idempotent_on_repeat(
        self,
        mock_collection,
    ):
        """Calling ensure_reports_indexes twice does not raise."""
        from app.db.collections.reports_collection import (
            ensure_reports_indexes,
        )

        asyncio.run(ensure_reports_indexes())
        asyncio.run(ensure_reports_indexes())

        assert mock_collection.create_index.await_count == 4
