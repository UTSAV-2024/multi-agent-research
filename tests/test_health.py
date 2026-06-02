import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestHealth:

    def test_health_endpoint(self):

        response = client.get("/health")

        assert response.status_code == 200

        body = response.json()

        assert body["status"] == "success"

        assert "message" in body

        assert "data" in body

        assert "environment" in body["data"]

        assert "version" in body["data"]

    def test_root_endpoint(self):

        response = client.get("/")

        assert response.status_code == 200

        body = response.json()

        assert body["status"] == "running"

        assert "message" in body
