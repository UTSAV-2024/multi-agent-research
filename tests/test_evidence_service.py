# ==========================================
# EVIDENCE SERVICE TESTS
# ==========================================
#
# Tests for retrieve_evidence() and its
# source diversity enforcement.
#
# ==========================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

from app.config.settings import settings


class TestRetrieveEvidence:
    """Tests for retrieve_evidence()."""

    @patch("app.services.evidence_service.retrieve_chunks")
    def test_returns_evidence_list(self, mock_retrieve):
        """retrieve_evidence returns a list of evidence dicts."""
        from app.services.evidence_service import retrieve_evidence

        mock_retrieve.return_value = {
            "ids": [["rep1:src1:1", "rep1:src1:2"]],
            "documents": [["Doc one.", "Doc two."]],
            "metadatas": [[
                {"chunk_id": 1, "url": "https://a.com"},
                {"chunk_id": 2, "url": "https://b.com"},
            ]],
            "distances": [[0.15, 0.42]],
        }

        result = retrieve_evidence(fact="test fact", top_k=2)

        assert isinstance(result, list)
        assert len(result) == 2

        # Check shape of each evidence dict
        for ev in result:
            assert "chunk_id" in ev
            assert "url" in ev
            assert "score" in ev
            assert isinstance(ev["chunk_id"], int)
            assert isinstance(ev["url"], str)
            assert isinstance(ev["score"], float)

    @patch("app.services.evidence_service.retrieve_chunks")
    def test_empty_results(self, mock_retrieve):
        """When no chunks returned, returns empty list."""
        from app.services.evidence_service import retrieve_evidence

        mock_retrieve.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        result = retrieve_evidence(fact="nothing", top_k=3)
        assert result == []

    @patch("app.services.evidence_service.retrieve_chunks")
    def test_empty_fact_returns_empty(self, mock_retrieve):
        """Empty fact string returns [] without calling retrieve_chunks."""
        from app.services.evidence_service import retrieve_evidence

        result = retrieve_evidence(fact="", top_k=3)
        assert result == []
        mock_retrieve.assert_not_called()

    @patch("app.services.evidence_service.retrieve_chunks")
    def test_retrieval_error_returns_empty(self, mock_retrieve):
        """When retrieve_chunks raises, returns [] gracefully."""
        from app.services.evidence_service import retrieve_evidence

        mock_retrieve.side_effect = RuntimeError("DB error")

        result = retrieve_evidence(fact="test", top_k=3)
        assert result == []

    @patch("app.services.evidence_service.retrieve_chunks")
    def test_whitespace_fact_returns_empty(self, mock_retrieve):
        """Whitespace-only fact returns [] without calling retrieve_chunks."""
        from app.services.evidence_service import retrieve_evidence

        result = retrieve_evidence(fact="   ", top_k=3)
        assert result == []
        mock_retrieve.assert_not_called()

    @patch("app.services.evidence_service.retrieve_chunks")
    def test_chunk_id_parsed_from_stable_id(self, mock_retrieve):
        """chunk_id is correctly parsed from the stable_id."""
        from app.services.evidence_service import retrieve_evidence

        mock_retrieve.return_value = {
            "ids": [["r:s:42"]],
            "documents": [["Content."]],
            "metadatas": [[
                {"chunk_id": 42, "url": "https://example.com"},
            ]],
            "distances": [[0.3]],
        }

        result = retrieve_evidence(fact="test", top_k=1)
        assert len(result) == 1
        assert result[0]["chunk_id"] == 42
        assert result[0]["url"] == "https://example.com"

    @patch("app.services.evidence_service.retrieve_chunks")
    def test_score_is_between_0_and_1(self, mock_retrieve):
        """Scores are normalised to [0, 1]."""
        from app.services.evidence_service import retrieve_evidence

        mock_retrieve.return_value = {
            "ids": [["r:s:1", "r:s:2"]],
            "documents": [["A.", "B."]],
            "metadatas": [[
                {"chunk_id": 1, "url": "https://a.com"},
                {"chunk_id": 2, "url": "https://b.com"},
            ]],
            "distances": [[0.1, 0.9]],
        }

        result = retrieve_evidence(fact="test", top_k=2)
        for ev in result:
            assert 0.0 <= ev["score"] <= 1.0


class TestSourceDiversity:
    """Tests for _enforce_source_diversity()."""

    def test_limits_per_source(self):
        """At most MAX_CHUNKS_PER_SOURCE items per source URL."""
        from app.services.evidence_service import _enforce_source_diversity

        items = [
            {"stable_id": "r:s:1", "metadata": {"url": "https://cnn.com/a"}},
            {"stable_id": "r:s:2", "metadata": {"url": "https://cnn.com/b"}},
            {"stable_id": "r:s:3", "metadata": {"url": "https://cnn.com/c"}},
            {"stable_id": "r:s:4", "metadata": {"url": "https://reuters.com/d"}},
            {"stable_id": "r:s:5", "metadata": {"url": "https://reuters.com/e"}},
        ]

        result = _enforce_source_diversity(items, max_per_source=2)

        # CNN should be capped at 2, Reuters at 2
        cnn_count = sum(
            1 for r in result if "cnn.com" in r["metadata"]["url"]
        )
        reuters_count = sum(
            1 for r in result if "reuters.com" in r["metadata"]["url"]
        )
        assert cnn_count <= 2
        assert reuters_count <= 2
        assert len(result) == 4  # 2 CNN + 2 Reuters

    def test_preserves_order(self):
        """Items preserve their original order within each source."""
        from app.services.evidence_service import _enforce_source_diversity

        items = [
            {"stable_id": "r:s:1", "metadata": {"url": "https://cnn.com/1"}},
            {"stable_id": "r:s:2", "metadata": {"url": "https://cnn.com/2"}},
            {"stable_id": "r:s:3", "metadata": {"url": "https://cnn.com/3"}},
        ]

        result = _enforce_source_diversity(items, max_per_source=2)

        assert len(result) == 2
        assert result[0]["stable_id"] == "r:s:1"
        assert result[1]["stable_id"] == "r:s:2"

    def test_no_url_kept(self):
        """Items without a URL are preserved."""
        from app.services.evidence_service import _enforce_source_diversity

        items = [
            {"stable_id": "r:s:1", "metadata": {"url": ""}},
            {"stable_id": "r:s:2", "metadata": {}},
        ]

        result = _enforce_source_diversity(items, max_per_source=2)
        assert len(result) == 2

    def test_zero_max_returns_empty(self):
        """max_per_source <= 0 returns empty list."""
        from app.services.evidence_service import _enforce_source_diversity

        items = [
            {"stable_id": "r:s:1", "metadata": {"url": "https://a.com"}},
        ]

        result = _enforce_source_diversity(items, max_per_source=0)
        assert result == []

    def test_below_limit_not_affected(self):
        """Items below the max per source are not affected."""
        from app.services.evidence_service import _enforce_source_diversity

        items = [
            {"stable_id": "r:s:1", "metadata": {"url": "https://a.com"}},
            {"stable_id": "r:s:2", "metadata": {"url": "https://b.com"}},
            {"stable_id": "r:s:3", "metadata": {"url": "https://c.com"}},
        ]

        result = _enforce_source_diversity(items, max_per_source=2)
        assert len(result) == 3

    def test_empty_items_list(self):
        """Empty input returns empty output."""
        from app.services.evidence_service import _enforce_source_diversity

        result = _enforce_source_diversity([], max_per_source=2)
        assert result == []


class TestSourceDiversityIntegration:
    """Integration: source diversity applied within retrieve_evidence."""

    @patch("app.services.evidence_service.retrieve_chunks")
    def test_diversity_applied_in_retrieve_evidence(self, mock_retrieve):
        """retrieve_evidence applies source diversity to results."""
        from app.services.evidence_service import retrieve_evidence

        # 4 chunks, 3 from same source
        mock_retrieve.return_value = {
            "ids": [["r:s:1", "r:s:2", "r:s:3", "r:s:4"]],
            "documents": [["A.", "B.", "C.", "D."]],
            "metadatas": [[
                {"chunk_id": 1, "url": "https://cnn.com/a"},
                {"chunk_id": 2, "url": "https://cnn.com/b"},
                {"chunk_id": 3, "url": "https://cnn.com/c"},
                {"chunk_id": 4, "url": "https://reuters.com/d"},
            ]],
            "distances": [[0.1, 0.2, 0.3, 0.4]],
        }

        result = retrieve_evidence(fact="test", top_k=4)

        # MAX_CHUNKS_PER_SOURCE=2 means at most 2 from CNN
        cnn_count = sum(1 for ev in result if "cnn.com" in ev["url"])
        assert cnn_count <= 2
        # Reuters should still be present
        reuters = [ev for ev in result if "reuters.com" in ev["url"]]
        assert len(reuters) == 1
