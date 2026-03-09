"""
conftest.py
-----------
Shared pytest fixtures used across the entire test suite.

Provides a realistic but synthetic travel dataset so that tests are
self-contained and do not depend on external files.
"""

import pandas as pd
import pytest


@pytest.fixture()
def sample_raw_df():
    """Return a small DataFrame mimicking the raw travel CSV."""
    data = {
        "Trip ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Destination": [
            "London, UK", "Phuket, Thailand", "Bali, Indonesia",
            "New York, USA", "Paris, France", "Tokyo, Japan",
            "Sydney, Australia", "Rome, Italy", "Dubai, UAE",
            "Barcelona, Spain",
        ],
        "Start date": [
            "5/1/2023", "6/15/2023", "7/1/2023", "8/15/2023",
            "9/1/2023", "10/1/2023", "11/1/2023", "12/1/2023",
            "1/15/2024", "2/1/2024",
        ],
        "End date": [
            "5/8/2023", "6/20/2023", "7/8/2023", "8/29/2023",
            "9/5/2023", "10/7/2023", "11/5/2023", "12/5/2023",
            "1/20/2024", "2/7/2024",
        ],
        "Duration (days)": [7, 5, 7, 14, 4, 6, 4, 4, 5, 6],
        "Traveler name": [
            "John Smith", "Jane Doe", "David Lee", "Sarah Johnson",
            "Alice Brown", "Bob Wilson", "Carol White", "Dan Green",
            "Eve Black", "Frank Grey",
        ],
        "Traveler age": [35, 28, 45, 29, 52, 31, 40, 25, 60, 38],
        "Traveler gender": [
            "Male", "Female", "Male", "Female", "Female",
            "Male", "Female", "Male", "Female", "Male",
        ],
        "Traveler nationality": [
            "American", "Canadian", "Korean", "British", "French",
            "German", "Australian", "Italian", "Emirati", "Spanish",
        ],
        "Accommodation type": [
            "Hotel", "Resort", "Villa", "Hotel", "Airbnb",
            "Hotel", "Resort", "Hotel", "Resort", "Villa",
        ],
        "Accommodation cost": [
            1200, 800, 1000, 2000, 600,
            1500, 900, 700, 1800, 1100,
        ],
        "Transportation type": [
            "Flight", "Flight", "Flight", "Flight", "Train",
            "Flight", "Flight", "Flight", "Flight", "Train",
        ],
        "Transportation cost": [
            600, 500, 700, 1000, 200,
            800, 650, 450, 900, 300,
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def sample_clean_df():
    """A small DataFrame that mimics post-cleaning state (after clean_dataframe)."""
    return pd.DataFrame({
        "destination": ["London, UK", "Bali, Indonesia", "Paris, France",
                        "Tokyo, Japan", "Sydney, Australia", "Rome, Italy",
                        "Dubai, UAE", "Barcelona, Spain", "Phuket, Thailand",
                        "New York, USA"],
        "start_date": ["5/1/2023", "7/1/2023", "9/1/2023", "10/1/2023",
                        "11/1/2023", "12/1/2023", "1/15/2024", "2/1/2024",
                        "6/15/2023", "8/15/2023"],
        "end_date": ["5/8/2023", "7/8/2023", "9/5/2023", "10/7/2023",
                      "11/5/2023", "12/5/2023", "1/20/2024", "2/7/2024",
                      "6/20/2023", "8/29/2023"],
        "duration_days": [7, 7, 4, 6, 4, 4, 5, 6, 5, 14],
        "traveler_age": [35, 45, 52, 31, 40, 25, 60, 38, 28, 29],
        "traveler_gender": ["Male", "Male", "Female", "Male", "Female",
                            "Male", "Female", "Male", "Female", "Female"],
        "traveler_nationality": ["American", "Korean", "French", "German",
                                  "Australian", "Italian", "Emirati",
                                  "Spanish", "Canadian", "British"],
        "accommodation_type": ["Hotel", "Villa", "Airbnb", "Hotel",
                                "Resort", "Hotel", "Resort", "Villa",
                                "Resort", "Hotel"],
        "accommodation_cost": [1200.0, 1000.0, 600.0, 1500.0, 900.0,
                                700.0, 1800.0, 1100.0, 800.0, 2000.0],
        "transportation_type": ["Flight", "Flight", "Train", "Flight",
                                 "Flight", "Flight", "Flight", "Train",
                                 "Flight", "Flight"],
        "transportation_cost": [600.0, 700.0, 200.0, 800.0, 650.0,
                                 450.0, 900.0, 300.0, 500.0, 1000.0],
        "total_cost": [1800.0, 1700.0, 800.0, 2300.0, 1550.0,
                        1150.0, 2700.0, 1400.0, 1300.0, 3000.0],
        "destination_city": ["London", "Bali", "Paris", "Tokyo",
                              "Sydney", "Rome", "Dubai", "Barcelona",
                              "Phuket", "New York"],
        "destination_country": ["UK", "Indonesia", "France", "Japan",
                                 "Australia", "Italy", "UAE", "Spain",
                                 "Thailand", "USA"],
        "travel_month": [5, 7, 9, 10, 11, 12, 1, 2, 6, 8],
        "day_of_week": [0, 5, 4, 6, 2, 4, 0, 3, 3, 1],
    })


@pytest.fixture()
def sample_feature_df():
    """Feature matrix matching the SETTINGS feature lists."""
    return pd.DataFrame({
        "duration_days": [7, 5, 7, 14, 4, 6, 4, 4, 5, 6],
        "traveler_age": [35, 28, 45, 29, 52, 31, 40, 25, 60, 38],
        "travel_month": [5, 6, 7, 8, 9, 10, 11, 12, 1, 2],
        "day_of_week": [0, 3, 5, 1, 4, 6, 2, 4, 0, 3],
        "destination_country": ["UK", "Thailand", "Indonesia", "USA",
                                 "France", "Japan", "Australia", "Italy",
                                 "UAE", "Spain"],
        "traveler_gender": ["Male", "Female", "Male", "Female", "Female",
                            "Male", "Female", "Male", "Female", "Male"],
        "traveler_nationality": ["American", "Canadian", "Korean", "British",
                                  "French", "German", "Australian", "Italian",
                                  "Emirati", "Spanish"],
        "accommodation_type": ["Hotel", "Resort", "Villa", "Hotel", "Airbnb",
                                "Hotel", "Resort", "Hotel", "Resort", "Villa"],
        "transportation_type": ["Flight", "Flight", "Flight", "Flight", "Train",
                                 "Flight", "Flight", "Flight", "Flight", "Train"],
    })


@pytest.fixture()
def sample_target():
    """Target vector matching the feature matrix."""
    return pd.Series(
        [1800, 1300, 1700, 3000, 800, 2300, 1550, 1150, 2700, 1400],
        name="total_cost",
    )


@pytest.fixture()
def sample_raw_csv(sample_raw_df, tmp_path):
    """Write the sample DataFrame as a CSV and return the path."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "travel_raw.csv"
    sample_raw_df.to_csv(path, index=False)
    return path