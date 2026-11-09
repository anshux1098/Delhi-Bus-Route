"""Validation benchmarks for the Delhi transit planner.

Compares planner ETAs against Google Maps transit or, when no API key is
configured, against the published static GTFS timetable.
"""

from src.evaluation.google_maps_client import (
    GtfsTimetable,
    fetch_google_maps_transit_eta,
    load_gtfs_timetable,
)

__all__ = [
    "GtfsTimetable",
    "fetch_google_maps_transit_eta",
    "load_gtfs_timetable",
]
