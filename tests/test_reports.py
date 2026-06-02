import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestReports:

    @patch(
        "app.api.routes.reports.get_report_by_id",
        new_callable=AsyncMock
    )
    def test_get_report_success(
        self,
        mock_get_report
    ):

        mock_get_report.return_value = {
            "_id": "507f1f77bcf86cd799439011",
            "topic": "Test Topic",
            "report": "Test report content"
        }

        response = client.get(
            "/reports/507f1f77bcf86cd799439011"
        )

        assert response.status_code == 200

        body = response.json()

        assert body["status"] == "success"

        assert body["data"]["_id"] == "507f1f77bcf86cd799439011"

        assert body["data"]["topic"] == "Test Topic"

    @patch(
        "app.api.routes.reports.get_report_by_id",
        new_callable=AsyncMock
    )
    def test_get_report_not_found(
        self,
        mock_get_report
    ):

        mock_get_report.return_value = None

        response = client.get(
            "/reports/000000000000000000000000"
        )

        assert response.status_code == 404

    @patch(
        "app.api.routes.reports.delete_report",
        new_callable=AsyncMock
    )
    def test_delete_report_success(
        self,
        mock_delete_report
    ):

        mock_delete_report.return_value = True

        response = client.request(
            "DELETE",
            "/reports/507f1f77bcf86cd799439011"
        )

        assert response.status_code == 200

        body = response.json()

        assert body["status"] == "success"

    @patch(
        "app.api.routes.reports.delete_report",
        new_callable=AsyncMock
    )
    def test_delete_report_not_found(
        self,
        mock_delete_report
    ):

        mock_delete_report.return_value = False

        response = client.request(
            "DELETE",
            "/reports/000000000000000000000000"
        )

        assert response.status_code == 404
