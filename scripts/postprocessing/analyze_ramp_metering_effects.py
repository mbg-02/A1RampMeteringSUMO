#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
from collections import defaultdict
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "sumo_plot_cache"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPLICATION_DIR_PATTERN = re.compile(r"rep_(\d+)_seed_(\d+)$")

ZURICH_EDGES = (
    "E1Z E2Z E3Z E4Z E5Z E6Z E7Z E8Z E9Z E10Z E11Z E12Z E13Z E14Z E15Z "
    "E16Z E17Z E18Z E19Z E20Z E21Z E22Z E23Z E24Z E25Z E26Z E27Z E28Z "
    "E29Z E30Z E31Z E32Z E33Z E34Z E35Z E36Z E37Z E38Z E39Z E40Z E41Z "
    "E42Z E43Z E44Z E45Z E46Z E47Z E48Z E49Z E50Z E51Z E52Z E53Z E54Z "
    "E55Z E56Z E57Z E58Z E59Z E60Z E61Z"
).split()
BERNE_EDGES = (
    "E1B E2B E3B E4B E5B E6B E7B E8B E9B E10B E11B E12B E13B E14B E15B "
    "E16B E17B E18B E19B E20B E21B E22B E23B E24B E25B E26B E27B E28B "
    "E29B E30B E31B E32B E33B E34B E35B E36B E37B E38B E39B E40B E41B "
    "E42B E43B E44B E45B E46B E47B E48B E49B E50B E51B E52B E53B E54B "
    "E55B E56B E57B E58B E59B E60B E61B E62B"
).split()
DIRECTION_EDGES = {
    "berne": BERNE_EDGES,
    "zurich": ZURICH_EDGES,
}
MAINLINE_EDGE_SET = set(BERNE_EDGES).union(ZURICH_EDGES)

RAMP_ORDER = (
    "48_B",
    "48_Z",
    "49_B",
    "49_Z",
    "50_B",
    "50_Z",
    "51_B",
    "51_Z",
    "52_B",
    "52_Z",
)
PLOT_RAMP_ORDER = tuple(
    sorted(RAMP_ORDER, key=lambda ramp_id: (ramp_id.split("_")[1], int(ramp_id.split("_")[0])))
)

TRIP_METRICS = (
    "duration_s",
    "time_loss_s",
    "waiting_time_s",
    "depart_delay_s",
)
CONTROLLER_METRICS = (
    "queue_length_m",
    "queue_ratio",
    "metering_rate",
    "metering_occupancy",
    "queue_override_active",
)


@dataclass(frozen=True)
class BaseRun:
    replication: int
    seed: int
    tripinfo_path: Path
    edgedata_path: Path


@dataclass(frozen=True)
class SignalRun:
    kp: str
    replication: int
    seed: int
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an uncontrolled validation scenario with ALINEA ramp metering. "
            "Outputs paired mainline, ramp-user, and controller summaries."
        )
    )
    parser.add_argument(
        "--base-dir",
        default="output sim/val_base_scenario",
        help="No-signal validation folder containing tripinfoN.xml and edgedataN.xml.",
    )
    parser.add_argument(
        "--signal-dir",
        default="output sim/alinea_run_astra_param_",
        help="ALINEA calibration/run folder containing runs/kp_*/rep_*_seed_*/.",
    )
    parser.add_argument(
        "--kp",
        help=(
            "Kp value to compare, for example 15. Defaults to the first row of "
            "alinea_stochastic_ranking.csv. Use 'all' to compare every kp folder."
        ),
    )
    parser.add_argument(
        "--det-rm",
        default="detRM.add.xml",
        help="Ramp-meter detector additional file used to identify metered ramp edges.",
    )
    parser.add_argument(
        "--net-file",
        default="network.net.xml",
        help="SUMO network file used to expand ramp approach edge prefixes.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for CSV and plot outputs. Defaults to "
            "<signal-dir>/ramp_metering_effects."
        ),
    )
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = (Path.cwd() / path, PROJECT_DIR / path, SCRIPT_DIR / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_DIR / path


def format_kp(value: str | float) -> str:
    if isinstance(value, str):
        value = value.strip()
        try:
            number = float(value)
        except ValueError:
            return value.removeprefix("kp_")
    else:
        number = float(value)

    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def output_dir_for(signal_dir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return resolve_path(output_dir_arg)
    return signal_dir / "ramp_metering_effects"


def load_base_runs(base_dir: Path) -> dict[int, BaseRun]:
    seeds_path = base_dir / "validation_run_seeds.csv"
    if not seeds_path.exists():
        raise FileNotFoundError(f"Missing seed mapping: {seeds_path}")

    runs_by_seed: dict[int, BaseRun] = {}
    with seeds_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            replication = int(row["run"])
            seed = int(row["seed"])
            tripinfo_path = base_dir / f"tripinfo{replication}.xml"
            edgedata_path = base_dir / f"edgedata{replication}.xml"
            if not tripinfo_path.exists() or not edgedata_path.exists():
                continue
            runs_by_seed[seed] = BaseRun(
                replication=replication,
                seed=seed,
                tripinfo_path=tripinfo_path,
                edgedata_path=edgedata_path,
            )
    if not runs_by_seed:
        raise FileNotFoundError(f"No complete base runs found in {base_dir}")
    return runs_by_seed


def best_kp_from_ranking(signal_dir: Path) -> str:
    ranking_path = signal_dir / "alinea_stochastic_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"Missing {ranking_path}; pass --kp explicitly or create the ranking first."
        )
    with ranking_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            return format_kp(row["kp"])
    raise ValueError(f"{ranking_path} does not contain any kp rows")


def discover_kps(signal_dir: Path, kp_arg: str | None) -> list[str]:
    runs_dir = signal_dir / "runs"
    if kp_arg is None:
        return [best_kp_from_ranking(signal_dir)]
    if kp_arg.lower() != "all":
        return [format_kp(kp_arg)]

    kps = []
    for path in sorted(runs_dir.glob("kp_*")):
        if path.is_dir():
            kps.append(format_kp(path.name.removeprefix("kp_")))
    if not kps:
        raise FileNotFoundError(f"No kp_* folders found in {runs_dir}")
    return kps


def load_signal_runs(signal_dir: Path, kps: list[str]) -> dict[tuple[str, int], SignalRun]:
    runs_by_kp_seed: dict[tuple[str, int], SignalRun] = {}
    for kp in kps:
        kp_dir = signal_dir / "runs" / f"kp_{kp}"
        if not kp_dir.exists():
            raise FileNotFoundError(f"Missing ALINEA kp folder: {kp_dir}")
        for run_dir in sorted(kp_dir.glob("rep_*_seed_*")):
            match = REPLICATION_DIR_PATTERN.fullmatch(run_dir.name)
            if match is None:
                continue
            replication = int(match.group(1))
            seed = int(match.group(2))
            if not (run_dir / "tripinfo.xml").exists() or not (
                run_dir / "edgedata.xml"
            ).exists():
                continue
            runs_by_kp_seed[(kp, seed)] = SignalRun(
                kp=kp,
                replication=replication,
                seed=seed,
                run_dir=run_dir,
            )
    if not runs_by_kp_seed:
        raise FileNotFoundError(f"No complete ALINEA runs found for {kps}")
    return runs_by_kp_seed


def edge_from_lane(lane_id: str) -> str:
    if "_" not in lane_id:
        return lane_id
    return lane_id.rsplit("_", 1)[0]


def ramp_id_from_detector(detector_id: str) -> str | None:
    parts = detector_id.split("_")
    if len(parts) < 4 or parts[2] != "ad" or not parts[3].startswith("met"):
        return None
    return f"{parts[0]}_{parts[1]}"


def ramp_prefix(edge_id: str) -> str:
    match = re.match(r"(.+?in)(?:\d.*)?$", edge_id)
    if match:
        return match.group(1)
    return edge_id


def load_all_network_edge_ids(net_file: Path) -> set[str]:
    edge_ids: set[str] = set()
    for _, edge in ET.iterparse(net_file, events=("end",)):
        if edge.tag == "edge":
            edge_id = edge.attrib.get("id")
            if edge_id and not edge_id.startswith(":"):
                edge_ids.add(edge_id)
        edge.clear()
    return edge_ids


def load_ramp_edges(det_rm_path: Path, net_file: Path) -> dict[str, set[str]]:
    all_edge_ids = load_all_network_edge_ids(net_file)
    seed_edges: dict[str, set[str]] = defaultdict(set)

    for _, detector in ET.iterparse(det_rm_path, events=("end",)):
        if detector.tag != "laneAreaDetector":
            detector.clear()
            continue
        ramp_id = ramp_id_from_detector(detector.attrib.get("id", ""))
        lane = detector.attrib.get("lane")
        if ramp_id is not None and lane:
            seed_edges[ramp_id].add(edge_from_lane(lane))
        detector.clear()

    ramp_edges: dict[str, set[str]] = {}
    for ramp_id, edges in seed_edges.items():
        expanded = set(edges)
        for edge_id in edges:
            prefix = ramp_prefix(edge_id)
            expanded.update(
                candidate
                for candidate in all_edge_ids
                if candidate == prefix or candidate.startswith(prefix)
            )
        ramp_edges[ramp_id] = expanded.difference(MAINLINE_EDGE_SET)

    missing = [ramp_id for ramp_id in RAMP_ORDER if ramp_id not in ramp_edges]
    if missing:
        raise ValueError(
            f"{det_rm_path} did not identify metered edge(s) for: {', '.join(missing)}"
        )
    return ramp_edges


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_sum(values: list[float]) -> float:
    return float(np.sum(values)) if values else float("nan")


def pct_change(delta: float, base: float) -> float:
    if not math.isfinite(delta) or not math.isfinite(base) or base == 0:
        return float("nan")
    return 100.0 * delta / base


def summarize_trip_values(values: dict[str, list[float]]) -> dict[str, float]:
    return {
        "count": float(len(values["duration_s"])),
        "mean_duration_s": safe_mean(values["duration_s"]),
        "mean_time_loss_s": safe_mean(values["time_loss_s"]),
        "mean_waiting_time_s": safe_mean(values["waiting_time_s"]),
        "mean_depart_delay_s": safe_mean(values["depart_delay_s"]),
        "sum_time_loss_s": safe_sum(values["time_loss_s"]),
    }


def empty_trip_values() -> dict[str, list[float]]:
    return {metric: [] for metric in TRIP_METRICS}


def parse_tripinfo_by_scope(
    tripinfo_path: Path,
    ramp_edges: dict[str, set[str]],
) -> dict[str, dict[str, float]]:
    edge_to_ramp = {
        edge_id: ramp_id
        for ramp_id, edge_ids in ramp_edges.items()
        for edge_id in edge_ids
    }
    values_by_scope: dict[str, dict[str, list[float]]] = defaultdict(empty_trip_values)

    for _, trip in ET.iterparse(tripinfo_path, events=("end",)):
        if trip.tag != "tripinfo":
            trip.clear()
            continue

        depart_lane = trip.attrib.get("departLane", "")
        depart_edge = edge_from_lane(depart_lane)
        duration = float(trip.attrib.get("duration", 0.0))
        time_loss = float(trip.attrib.get("timeLoss", 0.0))
        waiting_time = float(trip.attrib.get("waitingTime", 0.0))
        depart_delay = float(trip.attrib.get("departDelay", 0.0))

        scope_values = {
            "duration_s": duration,
            "time_loss_s": time_loss,
            "waiting_time_s": waiting_time,
            "depart_delay_s": depart_delay,
        }
        for metric, value in scope_values.items():
            values_by_scope["all_trips"][metric].append(value)

        if depart_edge in MAINLINE_EDGE_SET:
            for metric, value in scope_values.items():
                values_by_scope["mainline_departures"][metric].append(value)

        ramp_id = edge_to_ramp.get(depart_edge)
        if ramp_id is not None:
            for metric, value in scope_values.items():
                values_by_scope["all_metered_ramps"][metric].append(value)
                values_by_scope[f"ramp_{ramp_id}"][metric].append(value)

        trip.clear()

    return {
        scope: summarize_trip_values(values)
        for scope, values in sorted(values_by_scope.items())
    }


def parse_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_mainline_edgedata(edgedata_path: Path) -> dict[str, dict[str, float]]:
    per_direction: dict[str, dict[str, list[float]]] = {
        direction: defaultdict(list) for direction in DIRECTION_EDGES
    }

    for _, interval in ET.iterparse(edgedata_path, events=("end",)):
        if interval.tag != "interval":
            continue
        begin = parse_float(interval.attrib.get("begin"))
        end = parse_float(interval.attrib.get("end"))
        if begin is None or end is None:
            interval.clear()
            continue
        duration_h = max(end - begin, 0.0) / 3600.0

        edge_values = {edge.attrib.get("id"): edge for edge in interval.findall("edge")}
        for direction, edge_ids in DIRECTION_EDGES.items():
            corridor_traveltime = 0.0
            sampled_seconds = 0.0
            speed_weighted_sum = 0.0
            density_values = []
            total_time_loss = 0.0
            total_left = 0.0
            total_flow_weight = 0.0
            interval_has_data = False

            for edge_id in edge_ids:
                edge = edge_values.get(edge_id)
                if edge is None:
                    continue
                traveltime = parse_float(edge.attrib.get("traveltime"))
                speed = parse_float(edge.attrib.get("speed"))
                density = parse_float(edge.attrib.get("density"))
                sampled = parse_float(edge.attrib.get("sampledSeconds")) or 0.0
                time_loss = parse_float(edge.attrib.get("timeLoss"))
                left = parse_float(edge.attrib.get("left")) or 0.0
                entered = parse_float(edge.attrib.get("entered")) or 0.0
                flow = parse_float(edge.attrib.get("flow"))

                if traveltime is not None:
                    corridor_traveltime += traveltime
                    interval_has_data = True
                if speed is not None and sampled > 0:
                    speed_weighted_sum += speed * sampled
                    sampled_seconds += sampled
                if density is not None:
                    density_values.append(density)
                if time_loss is not None:
                    vehicle_count = max(left, entered)
                    total_time_loss += time_loss * vehicle_count
                    total_left += vehicle_count
                if flow is not None and duration_h > 0:
                    total_flow_weight += flow * duration_h

            if interval_has_data:
                per_direction[direction]["corridor_traveltime_s"].append(
                    corridor_traveltime
                )
            if sampled_seconds > 0:
                per_direction[direction]["weighted_speed_kmh"].append(
                    speed_weighted_sum / sampled_seconds * 3.6
                )
            if density_values:
                per_direction[direction]["mean_density_veh_per_km"].append(
                    float(np.mean(density_values))
                )
            if total_left > 0:
                per_direction[direction]["vehicle_weighted_time_loss_s"].append(
                    total_time_loss / total_left
                )
            if total_flow_weight > 0:
                per_direction[direction]["mean_flow_veh_per_h"].append(
                    total_flow_weight / max(duration_h, 1e-9) / len(edge_ids)
                )
        interval.clear()

    return {
        direction: {metric: safe_mean(values) for metric, values in metrics.items()}
        for direction, metrics in per_direction.items()
    }


def parse_ramp_edgedata(
    edgedata_path: Path,
    ramp_edges: dict[str, set[str]],
) -> dict[str, dict[str, float]]:
    edge_to_ramps: dict[str, list[str]] = defaultdict(list)
    for ramp_id, edge_ids in ramp_edges.items():
        for edge_id in edge_ids:
            edge_to_ramps[edge_id].append(ramp_id)

    accumulators: dict[str, dict[str, float]] = {
        ramp_id: defaultdict(float) for ramp_id in ramp_edges
    }
    for _, interval in ET.iterparse(edgedata_path, events=("end",)):
        if interval.tag != "interval":
            continue
        for edge in interval.findall("edge"):
            edge_id = edge.attrib.get("id")
            if edge_id not in edge_to_ramps:
                continue
            left = parse_float(edge.attrib.get("left")) or 0.0
            entered = parse_float(edge.attrib.get("entered")) or 0.0
            vehicle_count = max(left, entered)
            sampled = parse_float(edge.attrib.get("sampledSeconds")) or 0.0
            traveltime = parse_float(edge.attrib.get("traveltime"))
            time_loss = parse_float(edge.attrib.get("timeLoss"))
            waiting_time = parse_float(edge.attrib.get("waitingTime"))
            speed = parse_float(edge.attrib.get("speed"))
            occupancy = parse_float(edge.attrib.get("occupancy"))

            for ramp_id in edge_to_ramps[edge_id]:
                acc = accumulators[ramp_id]
                acc["vehicle_count"] += vehicle_count
                acc["sampled_seconds"] += sampled
                if traveltime is not None and vehicle_count > 0:
                    acc["traveltime_weighted_sum"] += traveltime * vehicle_count
                if time_loss is not None and vehicle_count > 0:
                    acc["time_loss_weighted_sum"] += time_loss * vehicle_count
                if waiting_time is not None and vehicle_count > 0:
                    acc["waiting_time_weighted_sum"] += waiting_time * vehicle_count
                if speed is not None and sampled > 0:
                    acc["speed_weighted_sum"] += speed * sampled
                if occupancy is not None and sampled > 0:
                    acc["occupancy_weighted_sum"] += occupancy * sampled
        interval.clear()

    results: dict[str, dict[str, float]] = {}
    for ramp_id, acc in accumulators.items():
        vehicle_count = acc["vehicle_count"]
        sampled_seconds = acc["sampled_seconds"]
        results[ramp_id] = {
            "edge_vehicle_count": vehicle_count,
            "edge_mean_traveltime_s": (
                acc["traveltime_weighted_sum"] / vehicle_count
                if vehicle_count > 0
                else float("nan")
            ),
            "edge_mean_time_loss_s": (
                acc["time_loss_weighted_sum"] / vehicle_count
                if vehicle_count > 0
                else float("nan")
            ),
            "edge_mean_waiting_time_s": (
                acc["waiting_time_weighted_sum"] / vehicle_count
                if vehicle_count > 0
                else float("nan")
            ),
            "edge_weighted_speed_kmh": (
                acc["speed_weighted_sum"] / sampled_seconds * 3.6
                if sampled_seconds > 0
                else float("nan")
            ),
            "edge_weighted_occupancy_pct": (
                acc["occupancy_weighted_sum"] / sampled_seconds
                if sampled_seconds > 0
                else float("nan")
            ),
        }
    return results


def paired_delta_row(
    row_prefix: Mapping[str, object],
    base_metrics: dict[str, float],
    signal_metrics: dict[str, float],
    metric_names: tuple[str, ...],
) -> dict[str, object]:
    row: dict[str, object] = dict(row_prefix)
    for metric in metric_names:
        base_value = base_metrics.get(metric, float("nan"))
        signal_value = signal_metrics.get(metric, float("nan"))
        delta = signal_value - base_value
        row[f"base_{metric}"] = base_value
        row[f"signal_{metric}"] = signal_value
        row[f"delta_{metric}"] = delta
        row[f"pct_delta_{metric}"] = pct_change(delta, base_value)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_numeric(rows: list[dict[str, object]], group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        for key, value in row.items():
            if key in group_keys or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                grouped[group][key].append(numeric)

    summary_rows: list[dict[str, object]] = []
    for group, metrics in sorted(grouped.items()):
        row = {key: value for key, value in zip(group_keys, group)}
        for metric, values in sorted(metrics.items()):
            array = np.asarray(values, dtype=float)
            row[f"{metric}_n"] = int(array.size)
            row[f"{metric}_mean"] = float(np.mean(array))
            row[f"{metric}_median"] = float(np.median(array))
            row[f"{metric}_stddev"] = (
                float(np.std(array, ddof=1)) if array.size > 1 else float("nan")
            )
        summary_rows.append(row)
    return summary_rows


def parse_controller_timeseries(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tl_id = row.get("tl_id", "")
            metric = row.get("metric", "")
            if metric not in CONTROLLER_METRICS:
                continue
            try:
                value = float(row.get("value", "nan"))
            except ValueError:
                continue
            if math.isfinite(value):
                ramp_id = tl_id.removeprefix("RMS_")
                values[ramp_id][metric].append(value)

    return {
        ramp_id: {metric: safe_mean(metric_values) for metric, metric_values in metrics.items()}
        for ramp_id, metrics in values.items()
    }


def collect_comparisons(
    base_runs: dict[int, BaseRun],
    signal_runs: dict[tuple[str, int], SignalRun],
    kps: list[str],
    ramp_edges: dict[str, set[str]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    mainline_rows: list[dict[str, object]] = []
    ramp_edge_rows: list[dict[str, object]] = []
    trip_rows: list[dict[str, object]] = []
    controller_rows: list[dict[str, object]] = []

    base_trip_cache: dict[int, dict[str, dict[str, float]]] = {}
    base_edge_cache: dict[int, dict[str, dict[str, float]]] = {}
    base_ramp_edge_cache: dict[int, dict[str, dict[str, float]]] = {}

    mainline_metric_names = (
        "corridor_traveltime_s",
        "weighted_speed_kmh",
        "mean_density_veh_per_km",
        "vehicle_weighted_time_loss_s",
        "mean_flow_veh_per_h",
    )
    trip_metric_names = (
        "count",
        "mean_duration_s",
        "mean_time_loss_s",
        "mean_waiting_time_s",
        "mean_depart_delay_s",
        "sum_time_loss_s",
    )
    ramp_edge_metric_names = (
        "edge_vehicle_count",
        "edge_mean_traveltime_s",
        "edge_mean_time_loss_s",
        "edge_mean_waiting_time_s",
        "edge_weighted_speed_kmh",
        "edge_weighted_occupancy_pct",
    )

    for kp in kps:
        for seed, base_run in sorted(base_runs.items(), key=lambda item: item[1].replication):
            signal_run = signal_runs.get((kp, seed))
            if signal_run is None:
                continue

            if seed not in base_trip_cache:
                base_trip_cache[seed] = parse_tripinfo_by_scope(
                    base_run.tripinfo_path,
                    ramp_edges,
                )
            if seed not in base_edge_cache:
                base_edge_cache[seed] = parse_mainline_edgedata(base_run.edgedata_path)
            if seed not in base_ramp_edge_cache:
                base_ramp_edge_cache[seed] = parse_ramp_edgedata(
                    base_run.edgedata_path,
                    ramp_edges,
                )

            signal_trip_metrics = parse_tripinfo_by_scope(
                signal_run.run_dir / "tripinfo.xml",
                ramp_edges,
            )
            signal_edge_metrics = parse_mainline_edgedata(signal_run.run_dir / "edgedata.xml")
            signal_ramp_edge_metrics = parse_ramp_edgedata(
                signal_run.run_dir / "edgedata.xml",
                ramp_edges,
            )
            controller_metrics = parse_controller_timeseries(
                signal_run.run_dir / "controller_timeseries.csv"
            )

            run_prefix = {
                "kp": kp,
                "base_replication": base_run.replication,
                "signal_replication": signal_run.replication,
                "seed": seed,
            }

            for direction in sorted(DIRECTION_EDGES):
                mainline_rows.append(
                    paired_delta_row(
                        {**run_prefix, "direction": direction},
                        base_edge_cache[seed].get(direction, {}),
                        signal_edge_metrics.get(direction, {}),
                        mainline_metric_names,
                    )
                )

            for ramp_id in RAMP_ORDER:
                ramp_edge_rows.append(
                    paired_delta_row(
                        {**run_prefix, "ramp_id": ramp_id},
                        base_ramp_edge_cache[seed].get(ramp_id, {}),
                        signal_ramp_edge_metrics.get(ramp_id, {}),
                        ramp_edge_metric_names,
                    )
                )

            scopes = ["all_trips", "mainline_departures", "all_metered_ramps"]
            scopes.extend(f"ramp_{ramp_id}" for ramp_id in RAMP_ORDER)
            for scope in scopes:
                if scope not in base_trip_cache[seed] and scope not in signal_trip_metrics:
                    continue
                trip_rows.append(
                    paired_delta_row(
                        {**run_prefix, "scope": scope},
                        base_trip_cache[seed].get(scope, {}),
                        signal_trip_metrics.get(scope, {}),
                        trip_metric_names,
                    )
                )

            for ramp_id, metrics in sorted(controller_metrics.items()):
                row: dict[str, object] = {**run_prefix, "ramp_id": ramp_id}
                for metric in CONTROLLER_METRICS:
                    row[metric] = metrics.get(metric, float("nan"))
                controller_rows.append(row)

    return mainline_rows, ramp_edge_rows, trip_rows, controller_rows


def finite_values(rows: list[dict[str, object]], key: str, selector: tuple[str, object] | None = None) -> list[float]:
    values: list[float] = []
    for row in rows:
        if selector is not None and row.get(selector[0]) != selector[1]:
            continue
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def plot_summary(
    output_path: Path,
    mainline_rows: list[dict[str, object]],
    ramp_edge_rows: list[dict[str, object]],
    trip_rows: list[dict[str, object]],
    controller_rows: list[dict[str, object]],
    image_format: str,
    dpi: int,
) -> None:
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
        }
    ):
        fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)

        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        for direction in ("B", "Z"):
            for ramp_id in PLOT_RAMP_ORDER:
                if ramp_id.split("_")[1] != direction:
                    continue
                labels.append(ramp_id.replace("_", " "))
                values.append(
                    safe_mean(
                        finite_values(
                            ramp_edge_rows,
                            "delta_edge_mean_time_loss_s",
                            ("ramp_id", ramp_id),
                        )
                    )
                )
                colors.append("#ca8a04")

        x_positions = np.arange(len(labels))
        bars = ax.bar(x_positions, values, color=colors)
        ax.axhline(0, color="#111827", linewidth=1.2)
        ax.set_ylabel("ALINEA - base [s]")
        ax.set_xticks(x_positions, labels)
        ax.tick_params(axis="x", rotation=45)
        finite_bar_values = [value for value in values if math.isfinite(value)]
        label_offset = (
            0.03 * (max(finite_bar_values) - min(finite_bar_values))
            if finite_bar_values
            else 1.0
        )
        if finite_bar_values:
            ax.set_ylim(
                min(finite_bar_values) - 2.5 * label_offset,
                max(finite_bar_values) + 2.5 * label_offset,
            )
        for bar, value in zip(bars, values):
            if not math.isfinite(value):
                continue
            x = bar.get_x() + bar.get_width() / 2
            if value >= 0:
                y = value + label_offset
                va = "bottom"
            else:
                y = value - label_offset
                va = "top"
            ax.text(
                x,
                y,
                f"{value:.1f}",
                ha="center",
                va=va,
                fontsize=13,
            )
        ax.legend(
            handles=[
                Rectangle((0, 0), 1, 1, color="#ca8a04", label="Ramp edge time loss"),
            ],
            frameon=False,
            loc="upper left",
        )

    fig.savefig(output_path, format=image_format, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    base_dir = resolve_path(args.base_dir)
    signal_dir = resolve_path(args.signal_dir)
    det_rm_path = resolve_path(args.det_rm)
    net_file = resolve_path(args.net_file)
    output_dir = output_dir_for(signal_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kps = discover_kps(signal_dir, args.kp)
    base_runs = load_base_runs(base_dir)
    signal_runs = load_signal_runs(signal_dir, kps)
    ramp_edges = load_ramp_edges(det_rm_path, net_file)

    mainline_rows, ramp_edge_rows, trip_rows, controller_rows = collect_comparisons(
        base_runs=base_runs,
        signal_runs=signal_runs,
        kps=kps,
        ramp_edges=ramp_edges,
    )
    if not mainline_rows and not trip_rows:
        raise ValueError("No paired runs found. Check that base and ALINEA seeds match.")

    mainline_summary = summarize_numeric(mainline_rows, ("kp", "direction"))
    ramp_edge_summary = summarize_numeric(ramp_edge_rows, ("kp", "ramp_id"))
    trip_summary = summarize_numeric(trip_rows, ("kp", "scope"))
    controller_summary = summarize_numeric(controller_rows, ("kp", "ramp_id"))

    write_csv(output_dir / "mainline_run_comparison.csv", mainline_rows)
    write_csv(output_dir / "mainline_summary.csv", mainline_summary)
    write_csv(output_dir / "ramp_edge_run_comparison.csv", ramp_edge_rows)
    write_csv(output_dir / "ramp_edge_summary.csv", ramp_edge_summary)
    write_csv(output_dir / "trip_scope_run_comparison.csv", trip_rows)
    write_csv(output_dir / "trip_scope_summary.csv", trip_summary)
    write_csv(output_dir / "controller_run_summary.csv", controller_rows)
    write_csv(output_dir / "controller_summary.csv", controller_summary)

    plot_summary(
        output_dir / f"ramp_metering_effects_summary.{args.format}",
        mainline_rows,
        ramp_edge_rows,
        trip_rows,
        controller_rows,
        args.format,
        args.dpi,
    )

    print(f"Compared kps: {', '.join(kps)}")
    print(f"Matched paired runs: {len(mainline_rows) // len(DIRECTION_EDGES)}")
    print(f"Wrote ramp metering effect analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
