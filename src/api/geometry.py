"""Route polyline geometry extraction for the FastAPI server.

The static index stores shapes in a local planar metre frame around Connaught
Place. This module converts those vertices back to WGS84 ``[lon, lat]`` and
slices the exact road polyline between two stops so MapLibre draws real road
curvature (Ring Road, flyovers, service lanes) instead of straight lines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.matching.shape_indexer import LAT0, LON0, M_PER_DEG_LAT, M_PER_DEG_LON

__all__ = [
    "extract_leg_geojson_coords",
    "planar_to_lonlat",
    "lonlat_to_planar",
]


def planar_to_lonlat(x_m: float | np.ndarray, y_m: float | np.ndarray) -> Any:
    """Convert planar metric coordinates around Connaught Place to WGS84."""
    lon = LON0 + np.asarray(x_m, dtype=np.float64) / M_PER_DEG_LON
    lat = LAT0 + np.asarray(y_m, dtype=np.float64) / M_PER_DEG_LAT
    return lon, lat


def lonlat_to_planar(lon: float | np.ndarray, lat: float | np.ndarray) -> Any:
    """Convert WGS84 coordinates to the static index's planar metre frame."""
    x_m = (np.asarray(lon, dtype=np.float64) - LON0) * M_PER_DEG_LON
    y_m = (np.asarray(lat, dtype=np.float64) - LAT0) * M_PER_DEG_LAT
    return x_m, y_m


def _resolve_route_id(
    static_index: Any, route_id: Any, transit_graph: Any
) -> str | None:
    """Accept a GTFS route id, falling back to a graph route index."""
    if route_id is None:
        return None

    text = str(route_id).strip()
    if text and text in static_index.route_to_shapes:
        return text

    if isinstance(route_id, (int, np.integer)) and transit_graph is not None:
        route_ids = getattr(transit_graph, "route_ids", None)
        if route_ids is not None and 0 <= int(route_id) < len(route_ids):
            return str(route_ids[int(route_id)])
    return text or None


def _forward_stop_positions(
    stop_ids: list[str], from_stop_id: str, to_stop_id: str
) -> tuple[int, int] | None:
    """Find the closest ordered occurrence of (from, to) on a route."""
    from_positions = [i for i, stop_id in enumerate(stop_ids) if stop_id == from_stop_id]
    to_positions = [i for i, stop_id in enumerate(stop_ids) if stop_id == to_stop_id]
    if not from_positions or not to_positions:
        return None

    best: tuple[int, int, int] | None = None
    for i in from_positions:
        for j in to_positions:
            if j > i:
                gap = j - i
                if best is None or gap < best[0]:
                    best = (gap, i, j)
    return None if best is None else (best[1], best[2])


def _point_at_distance(
    s_m: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    distance_m: float,
) -> tuple[float, float]:
    """Linearly interpolate a point at an arc-length distance along a shape."""
    distance_m = float(np.clip(distance_m, s_m[0], s_m[-1]))
    idx = int(np.searchsorted(s_m, distance_m, side="right")) - 1
    idx = min(max(idx, 0), len(s_m) - 2)

    segment_m = float(s_m[idx + 1] - s_m[idx])
    if segment_m <= 1e-9:
        return float(x_m[idx]), float(y_m[idx])

    t = (distance_m - float(s_m[idx])) / segment_m
    t = min(max(t, 0.0), 1.0)
    return (
        float(x_m[idx] + t * (x_m[idx + 1] - x_m[idx])),
        float(y_m[idx] + t * (y_m[idx + 1] - y_m[idx])),
    )


def _slice_shape_between(
    shape: Any, from_distance_m: float, to_distance_m: float
) -> list[tuple[float, float]]:
    """Return planar vertices from ``from_distance_m`` to ``to_distance_m``.

    Runs in O(log n + k) for a shape with ``n`` vertices and ``k`` output
    vertices, including interpolated endpoints exactly on the polyline.
    """
    s_m = np.asarray(shape.s_m, dtype=np.float64)
    x_m = np.asarray(shape.x_m, dtype=np.float64)
    y_m = np.asarray(shape.y_m, dtype=np.float64)
    if s_m.size < 2:
        return []

    reverse = to_distance_m < from_distance_m
    d0, d1 = (
        (float(to_distance_m), float(from_distance_m))
        if reverse
        else (float(from_distance_m), float(to_distance_m))
    )

    points: list[tuple[float, float]] = [_point_at_distance(s_m, x_m, y_m, d0)]

    # Include every original vertex strictly between the two stop projections.
    first_vertex = int(np.searchsorted(s_m, d0, side="right"))
    last_vertex = int(np.searchsorted(s_m, d1, side="left"))
    for idx in range(first_vertex, last_vertex):
        points.append((float(x_m[idx]), float(y_m[idx])))

    points.append(_point_at_distance(s_m, x_m, y_m, d1))

    # Collapse duplicates created when a stop projects exactly onto a vertex.
    compact: list[tuple[float, float]] = [points[0]]
    for point in points[1:]:
        prev = compact[-1]
        if abs(point[0] - prev[0]) > 1e-9 or abs(point[1] - prev[1]) > 1e-9:
            compact.append(point)

    if len(compact) < 2:
        compact.append(compact[0])
    if reverse:
        compact.reverse()
    return compact


def _straight_line(
    from_lon: float, from_lat: float, to_lon: float, to_lat: float
) -> list[list[float]]:
    return [[float(from_lon), float(from_lat)], [float(to_lon), float(to_lat)]]


def extract_leg_geojson_coords(
    static_index: Any,
    transit_graph: Any,
    route_id: Any,
    from_stop_id: str | None,
    to_stop_id: str | None,
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    mode: str,
) -> list[list[float]]:
    """Extract ``[lon, lat]`` geometry for one journey leg.

    Args:
        static_index: Loaded :class:`~src.matching.shape_indexer.StaticIndex`.
        transit_graph: Compiled transit graph, used to resolve integer route
            indices and as a graph-aware fallback.
        route_id: GTFS route id (or route index when an integer is supplied).
        from_stop_id: Boarding stop id for a BUS leg.
        to_stop_id: Alighting stop id for a BUS leg.
        from_lat/from_lon: Leg origin coordinates.
        to_lat/to_lon: Leg destination coordinates.
        mode: ``"WALK"`` or ``"BUS"``.

    Returns:
        Ordered ``[lon, lat]`` vertices. WALK legs are a two-point straight
        line. BUS legs follow the exact shape polyline, including all original
        road vertices between the stops' projected arc-length positions.
    """
    if str(mode).upper() != "BUS":
        return _straight_line(from_lon, from_lat, to_lon, to_lat)

    resolved_route_id = _resolve_route_id(static_index, route_id, transit_graph)
    if not resolved_route_id or not from_stop_id or not to_stop_id:
        return _straight_line(from_lon, from_lat, to_lon, to_lat)

    from_stop_id = str(from_stop_id)
    to_stop_id = str(to_stop_id)
    shape_ids = static_index.route_to_shapes.get(resolved_route_id) or []

    for shape_id in shape_ids:
        info = static_index.route_shape_info.get((resolved_route_id, shape_id))
        shape = static_index.shapes.get(shape_id)
        if info is None or shape is None:
            continue
        if np.asarray(shape.s_m).size < 2:
            continue

        positions = _forward_stop_positions(info.stop_ids, from_stop_id, to_stop_id)
        if positions is None:
            continue

        from_pos, to_pos = positions
        dists = np.asarray(info.stop_dists_m, dtype=np.float64)
        if dists.size <= max(from_pos, to_pos):
            continue

        planar = _slice_shape_between(shape, float(dists[from_pos]), float(dists[to_pos]))
        if len(planar) < 2:
            continue

        xs = np.fromiter((p[0] for p in planar), dtype=np.float64, count=len(planar))
        ys = np.fromiter((p[1] for p in planar), dtype=np.float64, count=len(planar))
        lons, lats = planar_to_lonlat(xs, ys)
        return [
            [float(lon), float(lat)]
            for lon, lat in zip(lons.tolist(), lats.tolist())
        ]

    # No usable shape: keep the response renderable with a direct segment.
    return _straight_line(from_lon, from_lat, to_lon, to_lat)
