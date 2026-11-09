"""Pydantic v2 response schemas for the Delhi transit routing API.

These models mirror the Hackathon Deliverable response contract. Geometry
coordinates are ``[lon, lat]`` pairs so they can be passed directly to
GeoJSON sources and MapLibre GL JS.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "LegSchema",
    "JourneyOptionSchema",
    "JourneyResponseSchema",
]


class LegSchema(BaseModel):
    """One walking or bus leg of a journey option."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["WALK", "BUS"] = Field(
        ..., description="Travel mode for this leg."
    )
    from_stop_id: str | None = Field(
        None, description="GTFS stop id, or null for a free-form origin."
    )
    from_stop_name: str = Field(..., description="Display name of the leg origin.")
    from_lat: float = Field(..., description="Origin latitude (WGS84).")
    from_lon: float = Field(..., description="Origin longitude (WGS84).")

    to_stop_id: str | None = Field(
        None, description="GTFS stop id, or null for a free-form destination."
    )
    to_stop_name: str = Field(..., description="Display name of the leg destination.")
    to_lat: float = Field(..., description="Destination latitude (WGS84).")
    to_lon: float = Field(..., description="Destination longitude (WGS84).")

    routes: list[str] = Field(
        default_factory=list,
        description="Route short names; empty for WALK, multiple for common lines.",
    )
    route_display: str = Field(
        ..., description='Human-readable instruction, e.g. "Walk 435m".'
    )
    is_common_line: bool = Field(
        False, description="True when several routes form one attractive set."
    )

    wait_time_mins: float = Field(0.0, ge=0.0, description="Expected wait in minutes.")
    in_vehicle_mins: float = Field(
        0.0, ge=0.0, description="Expected in-vehicle ride in minutes."
    )
    free_flow_in_vehicle_mins: float = Field(
        0.0, ge=0.0, description="Free-flow in-vehicle ride in minutes."
    )

    walk_distance_m: float = Field(
        0.0, ge=0.0, description="Walking distance in metres."
    )
    walk_duration_mins: float = Field(
        0.0, ge=0.0, description="Walking duration in minutes."
    )
    stops_count: int = Field(
        0, ge=0, description="Number of stops traversed on a bus leg."
    )
    live_bus_used: bool = Field(
        False, description="True when live vehicle positions informed this leg."
    )

    geometry: list[tuple[float, float]] = Field(
        ...,
        description="Ordered [lon, lat] vertices for MapLibre / GeoJSON.",
    )


class JourneyOptionSchema(BaseModel):
    """A complete origin-to-destination journey option."""

    model_config = ConfigDict(extra="forbid")

    option_id: int = Field(..., ge=0, description="Zero-based option rank.")
    profile_label: str = Field(
        "Fastest",
        description="Winning profile badge: 'Fastest', 'Most Reliable (Low P85)', or 'Least Walking / Direct'.",
    )
    summary: str = Field(..., description="Short human-readable summary.")
    departure_time_ist: str = Field(..., description="Departure time in IST.")
    arrival_time_ist: str = Field(..., description="Expected arrival time in IST.")
    p85_arrival_time_ist: str = Field(
        ..., description="85th-percentile arrival time in IST."
    )

    total_duration_mins: float = Field(
        ..., ge=0.0, description="Expected total journey duration in minutes."
    )
    p85_total_duration_mins: float = Field(
        ..., ge=0.0, description="85th-percentile total duration in minutes."
    )
    p85_buffer_mins: float = Field(
        ..., ge=0.0, description="Extra p85 buffer over the expected duration."
    )
    free_flow_duration_mins: float = Field(
        ..., ge=0.0, description="Free-flow journey duration in minutes."
    )
    traffic_delay_mins: float = Field(
        ..., ge=0.0, description="Expected traffic delay over free flow."
    )

    transfers: int = Field(..., ge=0, description="Number of transfers.")
    total_walk_mins: float = Field(..., ge=0.0)
    total_wait_mins: float = Field(..., ge=0.0)
    total_ride_mins: float = Field(..., ge=0.0)

    legs: list[LegSchema] = Field(..., description="Ordered journey legs.")


class JourneyResponseSchema(BaseModel):
    """Top-level response returned by the journey planning endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: dict[str, Any] = Field(..., description="Normalised query parameters.")
    routing_latency_ms: float = Field(
        ..., ge=0.0, description="Server-side routing latency in milliseconds."
    )
    active_live_buses_count: int = Field(
        ..., ge=0, description="Live buses observed during this request."
    )
    options: list[JourneyOptionSchema] = Field(
        ..., description="Ranked journey options, best first."
    )
