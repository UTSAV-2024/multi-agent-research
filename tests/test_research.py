import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestResearch:

    def test_research_invalid_payload(self):

        response = client.post(
            "/research",
            json={}
        )

        assert response.status_code == 422

        body = response.json()

        assert body["status"] == "failed"

    def test_research_missing_topic(self):

        response = client.post(
            "/research",
            json={"topic": ""}
        )

        assert response.status_code == 422
