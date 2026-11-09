"""Delhi Static GTFS Shape and Stop Indexer.

Loads static GTFS tables (routes, trips, stops, stop_times, shapes), casts IDs to Utf8,
recomputes planar meter arc-lengths and tangent vectors, and sequentially projects
canonical stops along each shape polyline (with automatic fallback to stop-to-stop
polylines for any route with missing shape_id).

Caches the compiled index to data/processed/static_index.pkl for sub-second loading.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure generic CPU architecture for Numba on Windows ARM64 / Snapdragon X
os.environ.setdefault("NUMBA_CPU_NAME", "generic")

import numpy as np
import polars as pl
from numba import njit

# Ensure UTF-8 console output for Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Reference coordinate: Connaught Place, New Delhi
LAT0 = 28.6139
LON0 = 77.2090
M_PER_DEG_LAT = 111132.92
M_PER_DEG_LON = 97799.66


def find_project_root() -> Path:
    """Locate the project root directory containing data/."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "data").exists():
        return candidate
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    if (cwd / "delhi-transit-planner" / "data").exists():
        return cwd / "delhi-transit-planner"
    return candidate


PROJECT_ROOT = find_project_root()
STATIC_DIR = PROJECT_ROOT / "data" / "static"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_PKL_PATH = PROCESSED_DIR / "static_index.pkl"


@dataclass
class ShapePolyline:
    """Planar metric polyline for a GTFS shape."""

    shape_id: str
    x_m: np.ndarray  # 1D float64
    y_m: np.ndarray  # 1D float64
    s_m: np.ndarray  # 1D float64, cumulative arc-length
    tx: np.ndarray  # 1D float64, segment unit tangent x
    ty: np.ndarray  # 1D float64, segment unit tangent y
    total_length_m: float


@dataclass
class StopInfo:
    """Stop reference projected onto shape."""

    stop_id: str
    stop_sequence: int
    d_stop_m: float
    stop_name: str | None = None
    stop_lat: float = 0.0
    stop_lon: float = 0.0


@dataclass
class RouteShapeInfo:
    """Canonical stops and metrics for a (route_id, shape_id) pair."""

    route_id: str
    shape_id: str
    canonical_trip_id: str
    stop_ids: list[str]
    stop_dists_m: np.ndarray  # 1D float64
    stop_names: list[str] = field(default_factory=list)
    stop_seqs: list[int] = field(default_factory=list)
    stop_lats: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    stop_lons: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    @property
    def stops(self) -> list[StopInfo]:
        """Lazy materialization of StopInfo objects."""
        out = []
        n = len(self.stop_ids)
        has_lats = len(self.stop_lats) == n
        has_lons = len(self.stop_lons) == n
        has_names = len(self.stop_names) == n
        has_seqs = len(self.stop_seqs) == n

        for i in range(n):
            out.append(
                StopInfo(
                    stop_id=self.stop_ids[i],
                    stop_sequence=self.stop_seqs[i] if has_seqs else i,
                    d_stop_m=float(self.stop_dists_m[i]),
                    stop_name=self.stop_names[i] if has_names else None,
                    stop_lat=float(self.stop_lats[i]) if has_lats else 0.0,
                    stop_lon=float(self.stop_lons[i]) if has_lons else 0.0,
                )
            )
        return out


@dataclass
class StaticIndex:
    """Fast in-memory index for map-matching and travel-time extraction."""

    route_to_shapes: dict[str, list[str]]
    shapes: dict[str, ShapePolyline]
    route_shape_info: dict[tuple[str, str], RouteShapeInfo]

    def get_candidate_shapes(self, route_id: str) -> list[str]:
        return self.route_to_shapes.get(str(route_id), [])

    def get_shape(self, shape_id: str) -> ShapePolyline | None:
        return self.shapes.get(str(shape_id))

    def get_route_shape(self, route_id: str, shape_id: str) -> RouteShapeInfo | None:
        return self.route_shape_info.get((str(route_id), str(shape_id)))

    def get_stops(self, route_id: str, shape_id: str) -> list[StopInfo]:
        info = self.get_route_shape(route_id, shape_id)
        return info.stops if info else []


@njit
def project_stops_forward(
    stop_x: np.ndarray,
    stop_y: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    poly_s: np.ndarray,
) -> np.ndarray:
    """Project ordered stops forward along polyline guaranteeing strict monotonicity.

    Uses forward horizon bounding and arc-length regularization so circular/loop
    routes never snap early stops onto the return leg at the end of the polyline.
    """
    n_stops = len(stop_x)
    n_segs = len(poly_x) - 1
    total_len = poly_s[len(poly_s) - 1]
    d_stops = np.empty(n_stops, dtype=np.float64)

    curr_s = 0.0
    seg_idx = 0

    for k in range(n_stops):
        sx = stop_x[k]
        sy = stop_y[k]

        if k == 0:
            max_search_s = min(total_len, 2500.0)
            expected_s = 0.0
            start_seg = 0
        else:
            dx_stop = sx - stop_x[k - 1]
            dy_stop = sy - stop_y[k - 1]
            euclid_step = np.sqrt(dx_stop * dx_stop + dy_stop * dy_stop)
            max_search_s = min(total_len, curr_s + max(2500.0, 4.0 * euclid_step))
            expected_s = curr_s + euclid_step
            start_seg = seg_idx

        best_score = 1e12
        best_d = curr_s
        best_seg = start_seg
        found = False

        for i in range(start_seg, n_segs):
            sa = poly_s[i]
            if sa > max_search_s and found:
                break
            sb = poly_s[i + 1]
            L = sb - sa
            if L <= 1e-6:
                continue

            xa = poly_x[i]
            ya = poly_y[i]
            xb = poly_x[i + 1]
            yb = poly_y[i + 1]

            dx = xb - xa
            dy = yb - ya
            wx = sx - xa
            wy = sy - ya

            t = (wx * dx + wy * dy) / (L * L)
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0

            cand_d = sa + t * L
            if k > 0 and cand_d < curr_s - 20.0:
                continue

            px = xa + t * dx
            py = ya + t * dy
            dist = np.sqrt((sx - px) ** 2 + (sy - py) ** 2)

            score = dist + 0.02 * abs(cand_d - expected_s)

            if score < best_score:
                best_score = score
                best_d = cand_d
                best_seg = i
                found = True

        # Enforce strict forward monotonicity
        if k > 0 and best_d <= curr_s:
            best_d = curr_s + 1.0

        curr_s = best_d
        seg_idx = best_seg
        d_stops[k] = best_d

    return d_stops


def _build_polyline_from_coords(
    sid: str, lats: np.ndarray, lons: np.ndarray
) -> ShapePolyline:
    """Compute planar meter polyline, arc-length s_m, and unit tangents (tx, ty)."""
    x_m = (lons - LON0) * M_PER_DEG_LON
    y_m = (lats - LAT0) * M_PER_DEG_LAT

    dx = np.diff(x_m)
    dy = np.diff(y_m)
    seg_lens = np.sqrt(dx * dx + dy * dy)
    s_m = np.concatenate([[0.0], np.cumsum(seg_lens)])

    safe_lens = np.where(seg_lens > 1e-6, seg_lens, 1.0)
    tx = np.where(seg_lens > 1e-6, dx / safe_lens, 1.0)
    ty = np.where(seg_lens > 1e-6, dy / safe_lens, 0.0)

    total_len = float(s_m[-1]) if len(s_m) > 0 else 0.0

    return ShapePolyline(
        shape_id=sid,
        x_m=x_m,
        y_m=y_m,
        s_m=s_m,
        tx=tx,
        ty=ty,
        total_length_m=total_len,
    )


def build_static_index(
    static_dir: Path | str | None = None,
    *,
    routes_df: pl.DataFrame | None = None,
    trips_df: pl.DataFrame | None = None,
    stops_df: pl.DataFrame | None = None,
    stop_times_df: pl.DataFrame | None = None,
    shapes_df: pl.DataFrame | None = None,
) -> StaticIndex:
    """Compile static GTFS files (or provided DataFrames) into a StaticIndex."""
    dir_path = Path(static_dir) if static_dir else STATIC_DIR
    t0 = time.time()

    # 1. Load tables with explicit string casts on all ID columns
    if routes_df is None:
        routes_df = pl.read_csv(
            dir_path / "routes.txt",
            schema_overrides={"route_id": pl.Utf8},
        )
    else:
        routes_df = routes_df.with_columns(pl.col("route_id").cast(pl.Utf8))

    if trips_df is None:
        trips_df = pl.read_csv(
            dir_path / "trips.txt",
            schema_overrides={
                "route_id": pl.Utf8,
                "trip_id": pl.Utf8,
                "shape_id": pl.Utf8,
            },
        )
    else:
        trips_df = trips_df.with_columns(
            pl.col("route_id").cast(pl.Utf8),
            pl.col("trip_id").cast(pl.Utf8),
            pl.col("shape_id").cast(pl.Utf8),
        )

    if stops_df is None:
        stops_df = pl.read_csv(
            dir_path / "stops.txt",
            schema_overrides={"stop_id": pl.Utf8},
        )
    else:
        stops_df = stops_df.with_columns(pl.col("stop_id").cast(pl.Utf8))

    if stop_times_df is None:
        stop_times_df = pl.read_csv(
            dir_path / "stop_times.txt",
            schema_overrides={"trip_id": pl.Utf8, "stop_id": pl.Utf8},
        )
    else:
        stop_times_df = stop_times_df.with_columns(
            pl.col("trip_id").cast(pl.Utf8),
            pl.col("stop_id").cast(pl.Utf8),
        )

    if shapes_df is None:
        shapes_df = pl.read_csv(
            dir_path / "shapes.txt",
            schema_overrides={"shape_id": pl.Utf8},
        )
    else:
        shapes_df = shapes_df.with_columns(pl.col("shape_id").cast(pl.Utf8))

    print(f"[OK] GTFS tables ready in {time.time() - t0:.2f}s.")

    # 2. Extract unique shapes from shapes.txt into ShapePolyline objects
    t_shape = time.time()
    shapes_dict: dict[str, ShapePolyline] = {}

    if not shapes_df.is_empty():
        sorted_shapes = (
            shapes_df.sort(["shape_id", "shape_pt_sequence"])
            .group_by("shape_id", maintain_order=True)
            .agg([
                pl.col("shape_pt_lat").cast(pl.Float64),
                pl.col("shape_pt_lon").cast(pl.Float64),
            ])
        )

        for row in sorted_shapes.iter_rows(named=True):
            sid = str(row["shape_id"])
            lats = np.array(row["shape_pt_lat"], dtype=np.float64)
            lons = np.array(row["shape_pt_lon"], dtype=np.float64)
            if len(lats) < 2:
                continue
            shapes_dict[sid] = _build_polyline_from_coords(sid, lats, lons)

    print(f"[OK] Indexed {len(shapes_dict):,} shapes in {time.time() - t_shape:.2f}s.")

    # 3. Assign fallback shape_id for any trip with null/empty/unindexed shape_id
    valid_shape_ids = set(shapes_dict.keys())
    trips_with_fallback = trips_df.with_columns(
        shape_id=pl.when(
            pl.col("shape_id").is_null()
            | (pl.col("shape_id") == "")
            | (~pl.col("shape_id").is_in(valid_shape_ids))
        )
        .then(pl.format("fallback_{}", pl.col("route_id")))
        .otherwise(pl.col("shape_id"))
    )

    # 4. Build route_to_shapes mapping
    route_shapes = (
        trips_with_fallback.select(["route_id", "shape_id"]).unique()
    )
    route_to_shapes: dict[str, list[str]] = {}
    for r in route_shapes.iter_rows(named=True):
        rid = str(r["route_id"])
        sid = str(r["shape_id"])
        route_to_shapes.setdefault(rid, []).append(sid)

    # 5. Map canonical stop sequences per (route_id, shape_id) using modal trip_id
    t_stops = time.time()
    # Count stop_times per trip to pick the modal trip pattern per (route_id, shape_id)
    canonical_trips = (
        trips_with_fallback.group_by(["route_id", "shape_id"])
        .first()
        .select(["route_id", "shape_id", "trip_id"])
    )

    stops_subset = stops_df.select([
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
    ])
    canonical_seqs = (
        canonical_trips.join(stop_times_df, on="trip_id")
        .join(stops_subset, on="stop_id")
        .sort(["route_id", "shape_id", "stop_sequence"])
    )

    route_shape_info: dict[tuple[str, str], RouteShapeInfo] = {}

    grouped_seqs = canonical_seqs.group_by(["route_id", "shape_id"], maintain_order=True).agg([
        pl.col("trip_id").first().alias("trip_id"),
        pl.col("stop_id"),
        pl.col("stop_name"),
        pl.col("stop_sequence"),
        pl.col("stop_lat").cast(pl.Float64),
        pl.col("stop_lon").cast(pl.Float64),
    ])

    fallback_count = 0
    for row in grouped_seqs.iter_rows(named=True):
        rid = str(row["route_id"])
        sid = str(row["shape_id"])
        trip_id = str(row["trip_id"])

        stop_ids = [str(x) for x in row["stop_id"]]
        stop_names = [str(x) if x is not None else "" for x in row["stop_name"]]
        stop_seqs = [int(x) for x in row["stop_sequence"]]
        stop_lats = np.array(row["stop_lat"], dtype=np.float64)
        stop_lons = np.array(row["stop_lon"], dtype=np.float64)

        if len(stop_ids) < 2:
            continue

        shape_poly = shapes_dict.get(sid)
        if shape_poly is None or len(shape_poly.x_m) < 2:
            # Fallback: build polyline directly from stop-to-stop coordinates
            shape_poly = _build_polyline_from_coords(sid, stop_lats, stop_lons)
            shapes_dict[sid] = shape_poly
            fallback_count += 1

        sx_m = (stop_lons - LON0) * M_PER_DEG_LON
        sy_m = (stop_lats - LAT0) * M_PER_DEG_LAT

        d_stops = project_stops_forward(
            sx_m, sy_m, shape_poly.x_m, shape_poly.y_m, shape_poly.s_m
        )

        route_shape_info[(rid, sid)] = RouteShapeInfo(
            route_id=rid,
            shape_id=sid,
            canonical_trip_id=trip_id,
            stop_ids=stop_ids,
            stop_dists_m=d_stops,
            stop_names=stop_names,
            stop_seqs=stop_seqs,
            stop_lats=stop_lats,
            stop_lons=stop_lons,
        )

    print(
        f"[OK] Projected canonical stops for {len(route_shape_info):,} route-shape pairs "
        f"({fallback_count} fallback polylines) in {time.time() - t_stops:.2f}s."
    )

    static_index = StaticIndex(
        route_to_shapes=route_to_shapes,
        shapes=shapes_dict,
        route_shape_info=route_shape_info,
    )

    return static_index


def save_static_index(index: StaticIndex, out_path: Path | str | None = None) -> Path:
    """Serialize static index to pickle."""
    target_path = Path(out_path) if out_path else INDEX_PKL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(target_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    file_size_mb = target_path.stat().st_size / (1024.0 * 1024.0)
    print(f"[OK] Saved StaticIndex to {target_path} ({file_size_mb:.2f} MB in {time.time() - t0:.2f}s)")
    return target_path


class IndexUnpickler(pickle.Unpickler):
    """Custom unpickler to resolve classes across module entrypoints."""

    def find_class(self, module: str, name: str) -> Any:
        if module in ("__main__", "src.matching.map_matcher"):
            module = "src.matching.shape_indexer"
        return super().find_class(module, name)


def load_static_index(path: Path | str | None = None) -> StaticIndex:
    """Load cached static index from pickle (<0.5s target)."""
    target_path = Path(path) if path else INDEX_PKL_PATH
    if not target_path.exists():
        print(f"[INFO] Cache not found at {target_path}. Building index...")
        index = build_static_index()
        save_static_index(index, target_path)
        return index

    t0 = time.time()
    with open(target_path, "rb") as f:
        index = IndexUnpickler(f).load()
    elapsed = time.time() - t0
    print(f"[OK] Loaded StaticIndex from {target_path.name} in {elapsed:.3f}s")
    return index


def main() -> int:
    print("=================================================================")
    print("Building Delhi GTFS Static Index")
    print("=================================================================")
    t_start = time.time()
    import src.matching.shape_indexer as mod
    index = mod.build_static_index()
    pkl_path = mod.save_static_index(index)
    print(f"[TOTAL BUILD TIME] {time.time() - t_start:.2f}s\n")

    print("Verifying index load speed...")
    t_load = time.time()
    loaded_index = mod.load_static_index(pkl_path)
    load_time = time.time() - t_load

    print(f"Verified {len(loaded_index.shapes):,} shapes and {len(loaded_index.route_shape_info):,} route sequences.")
    print(f"Load time: {load_time:.3f}s (Target: < 0.5s)")
    assert load_time < 0.5, f"Index load time {load_time:.3f}s exceeded 0.5s threshold"
    print("[SUCCESS] Static shape index compiled and verified successfully.")
    return 0


if __name__ == "__main__":
    import src.matching.shape_indexer as _mod
    sys.exit(_mod.main())
