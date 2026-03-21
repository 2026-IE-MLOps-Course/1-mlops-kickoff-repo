# tests/test_api.py
"""Tests for the FastAPI prediction API."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


def test_travel_record_schema_rejects_extra_fields():
    """Pydantic should reject unknown fields (extra='forbid')."""
    from src.api import TravelRecord
    with pytest.raises(Exception):
        TravelRecord(
            destination="Paris, France",
            start_date="2024-06-01",
            end_date="2024-06-05",
            duration_days=4,
            traveler_age=30,
            traveler_gender="Male",
            traveler_nationality="US",
            accommodation_type="Hotel",
            accommodation_cost=500.0,
            transportation_type="Flight",
            transportation_cost=300.0,
            unknown_field="should_fail",
        )


def test_travel_record_schema_accepts_valid():
    """Pydantic should accept well-formed records."""
    from src.api import TravelRecord
    record = TravelRecord(
        destination="Paris, France",
        start_date="2024-06-01",
        end_date="2024-06-05",
        duration_days=4,
        traveler_age=30,
        traveler_gender="Male",
        traveler_nationality="US",
        accommodation_type="Hotel",
        accommodation_cost=500.0,
        transportation_type="Flight",
        transportation_cost=300.0,
    )
    assert record.duration_days == 4


def test_predict_request_wraps_records():
    """PredictRequest should hold a list of TravelRecords."""
    from src.api import TravelRecord, PredictRequest
    record = TravelRecord(
        destination="Tokyo, Japan",
        start_date="2024-07-01",
        end_date="2024-07-10",
        duration_days=9,
        traveler_age=25,
        traveler_gender="Female",
        traveler_nationality="UK",
        accommodation_type="Hostel",
        accommodation_cost=200.0,
        transportation_type="Train",
        transportation_cost=150.0,
    )
    req = PredictRequest(records=[record])
    assert len(req.records) == 1


def test_health_endpoint():
    """Test that /health returns status ok."""
    from fastapi.testclient import TestClient
    from src.api import app

    # Mock the lifespan to avoid loading a real model
    app.state.model = MagicMock()
    app.state.config = {}
    app.state.feature_columns = []

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
