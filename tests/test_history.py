import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestHistory:

    @patch(
        "app.api.routes.history.get_history",
        new_callable=AsyncMock
    )
    def test_history_endpoint(
        self,
        mock_get_history
    ):

        mock_get_history.return_value = []

        response = client.get("/history")

        assert response.status_code == 200

        body = response.json()

        assert body["status"] == "success"

        assert "count" in body

        assert "limit" in body

        assert "skip" in body

        assert "data" in body

    @patch(
        "app.api.routes.history.get_history",
        new_callable=AsyncMock
    )
    def test_history_with_pagination(
        self,
        mock_get_history
    ):

        mock_get_history.return_value = []

        response = client.get(
            "/history?limit=10&skip=5"
        )

        assert response.status_code == 200

        body = response.json()

        assert body["limit"] == 10

        assert body["skip"] == 5
