#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import csv
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import NormalDist
import subprocess
import sys
import tempfile
from typing import TypedDict
import xml.etree.ElementTree as ET

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output sim" / "alinea_calibration"

SUMO_HOME = Path(os.environ.get("SUMO_HOME", Path.home() / "sumo"))
sys.path.insert(0, str(SUMO_HOME / "tools"))
sys.path.insert(0, str(ROOT_DIR))

try:
    import traci  # pyright: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "Could not import SUMO TraCI (`traci`). Install SUMO and set SUMO_HOME "
        f"to the SUMO installation directory. Looked for TraCI under {SUMO_HOME / 'tools'}."
    ) from exc

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "alinea_calibration_mplconfig"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sumoITScontrol import RampMeter
from sumoITScontrol.control.ramp_metering import ALINEA


class RampConfig(TypedDict):
    tl_id: str
    mainline_sensors: list[str]
    queue_sensors: list[str]
    queue_override_sensors: list[str]


RAMP_CONFIGS: list[RampConfig] = [
    {
        "tl_id": "RMS_48_B",
        "mainline_sensors": ["48_B_ld_bf1", "48_B_ld_bf2"],
        "queue_sensors": ["48_B_ad_met", "48_B_ad_met2", "48_B_ad_met3"],
        "queue_override_sensors": ["48_B_ld_spillback1", "48_B_ld_spillback2"],
    },
    {
        "tl_id": "RMS_48_Z",
        "mainline_sensors": ["48_Z_ld_bf3", "48_Z_ld_bf4"],
        "queue_sensors": ["48_Z_ad_met", "48_Z_ad_met2", "48_Z_ad_met3"],
        "queue_override_sensors": ["48_Z_ld_spillback1", "48_Z_ld_spillback2"],
    },
    {
        "tl_id": "RMS_49_B",
        "mainline_sensors": ["49_B_ld_bf1", "49_B_ld_bf2"],
        "queue_sensors": ["49_B_ad_met", "49_B_ad_met2"],
        "queue_override_sensors": ["49_B_ld_spillback"],
    },
    {
        "tl_id": "RMS_49_Z",
        "mainline_sensors": ["49_Z_ld_bf3", "49_Z_ld_bf4"],
        "queue_sensors": ["49_Z_ad_met1", "49_Z_ad_met2"],
        "queue_override_sensors": ["49_Z_ld_spillback"],
    },
    {
        "tl_id": "RMS_50_B",
        "mainline_sensors": ["50_B_ld_bf1", "50_B_ld_bf2"],
        "queue_sensors": [
            "50_B_ad_met",
            "50_B_ad_met2",
            "50_B_ad_met3",
            "50_B_ad_met4",
            "50_B_ad_met5",
            "50_B_ad_met6",
        ],
        "queue_override_sensors": ["50_B_ld_spillback1", "50_B_ld_spillback2"],
    },
    {
        "tl_id": "RMS_50_Z",
        "mainline_sensors": ["50_Z_ld_bf3", "50_Z_ld_bf4"],
        "queue_sensors": [
            "50_Z_ad_met",
            "50_Z_ad_met2",
            "50_Z_ad_met3",
            "50_Z_ad_met4",
            "50_Z_ad_met5",
            "50_Z_ad_met6",
        ],
        "queue_override_sensors": ["50_Z_ld_spillback1", "50_Z_ld_spillback2"],
    },
    {
        "tl_id": "RMS_51_B",
        "mainline_sensors": ["51_B_ld_bf1", "51_B_ld_bf2"],
        "queue_sensors": ["51_B_ad_met", "51_B_ad_met2"],
        "queue_override_sensors": ["51_B_ld_spillback"],
    },
    {
        "tl_id": "RMS_51_Z",
        "mainline_sensors": ["51_Z_ld_bf3", "51_Z_ld_bf4"],
        "queue_sensors": ["51_Z_ad_met"],
        "queue_override_sensors": ["51_Z_ld_spillback"],
    },
    {
        "tl_id": "RMS_52_B",
        "mainline_sensors": ["52_B_ld_bf1", "52_B_ld_bf2"],
        "queue_sensors": ["52_B_ad_met", "52_B_ad_met2"],
        "queue_override_sensors": ["52_B_ld_spillback"],
    },
    {
        "tl_id": "RMS_52_Z",
        "mainline_sensors": ["52_Z_ld_bf3", "52_Z_ld_bf4"],
        "queue_sensors": ["52_Z_ad_met"],
        "queue_override_sensors": ["52_Z_ld_spillback"],
    },
]

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

SUMMARY_METRICS = (
    "objective",
    "avg_queue_ratio",
    "avg_normalized_occupancy_violation",
    "spillback_active_share",
    "avg_queue_length_m",
    "avg_occupancy_violation",
    "mean_trip_duration_s",
    "mean_trip_time_loss_s",
    "mean_trip_waiting_time_s",
    "mean_trip_depart_delay_s",
    "sum_trip_time_loss_s",
)
TIMESERIES_METRICS = (
    ("metering_rate", "Metering rate [%]"),
    ("metering_occupancy", "Mainline Occupancy [%]"),
    ("queue_length_m", "Queue length [m]"),
    ("queue_ratio", "Queue ratio [-]"),
    ("queue_occupancy", "Queue detector occupancy [%]"),
    ("queue_override_active", "Queue override active"),
)
CONTROLLER_AXIS_LABEL_FONTSIZE = 16
CONTROLLER_TICK_FONTSIZE = 13
CONTROLLER_LEGEND_FONTSIZE = 13
CONTROLLER_MEDIAN_LEGEND_FONTSIZE = 11
CONTROLLER_SUBPLOT_TITLE_FONTSIZE = 13
CONTROLLER_MEDIAN_FIGSIZE = (5.4, 5.0)
CONTROLLER_COLORS = (
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#7A3E9D",
    "#56B4E9",
)
CONTROLLER_ROLLING_MEDIAN_WINDOW_SEC = 900.0
RAW_RESULTS_FIELDNAMES = [
    "kp",
    "replication",
    "seed",
    "objective",
    "avg_queue_ratio",
    "avg_normalized_occupancy_violation",
    "spillback_active_share",
    "avg_queue_length_m",
    "avg_occupancy_violation",
    "trip_count",
    "mean_trip_duration_s",
    "mean_trip_time_loss_s",
    "mean_trip_waiting_time_s",
    "mean_trip_depart_delay_s",
    "sum_trip_time_loss_s",
    "queue_observation_count",
    "occupancy_observation_count",
]


class RunPlotRow(TypedDict):
    kp: float
    replication: int
    seed: int
    timeseries: dict[str, dict[str, list[list[float]]]]


class ReplicationResult(TypedDict):
    kp: float
    replication: int
    seed: int
    metrics: dict[str, float]
    timeseries: dict[str, dict[str, list[list[float]]]]


class ControllerLineStyle(TypedDict):
    color: str


RUN_DIR_PATTERN = re.compile(r"^rep_(\d+)_seed_(\d+)$")


def controller_line_style(index: int) -> ControllerLineStyle:
    return {
        "color": CONTROLLER_COLORS[index % len(CONTROLLER_COLORS)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate ALINEA K_P stochastically by reusing the same seed set for "
            "every candidate and reporting mean performance together with variance."
        )
    )
    parser.add_argument(
        "--sumo-config",
        default=str(ROOT_DIR / "A1.sumocfg"),
        help="SUMO configuration file.",
    )
    parser.add_argument(
        "--sumo-binary",
        default="sumo",
        help="SUMO binary to launch. Use sumo-gui for visual debugging.",
    )
    parser.add_argument(
        "--traci-port",
        type=int,
        help="Optional explicit TraCI port. Useful in restricted environments.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for raw results, summaries, plots, and metadata.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        help=(
            "Optional simulation horizon in seconds. If omitted, the simulation "
            "runs until no more vehicles are expected."
        ),
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.5,
        help="SUMO simulation step length in seconds.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=20,
        help="Number of seeds to evaluate per K_P candidate.",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help=(
            "Maximum number of replications to run in parallel for each K_P value. "
            "Use 10 to run 10 replications at once."
        ),
    )
    parser.add_argument(
        "--seed-generator-seed",
        type=int,
        default=20260526,
        help="Seed used to generate the documented SUMO seed list.",
    )
    parser.add_argument(
        "--seeds",
        help="Optional comma-separated seed list. Overrides --replications generation.",
    )
    parser.add_argument(
        "--kp-min",
        type=float,
        default=5.0,
        help="Lower bound of the K_P search space.",
    )
    parser.add_argument(
        "--kp-max",
        type=float,
        default=50.0,
        help="Upper bound of the K_P search space.",
    )
    parser.add_argument(
        "--kp-step",
        type=float,
        default=5.0,
        help="Grid-search step size for K_P.",
    )
    parser.add_argument(
        "--target-occupancy",
        type=float,
        default=10.0,
        help="Target occupancy for all ALINEA controllers.",
    )
    parser.add_argument(
        "--cycle-duration",
        type=int,
        default=60,
        help="Ramp metering cycle duration in seconds.",
    )
    parser.add_argument(
        "--measurement-period-sec",
        type=float,
        default=300.0,
        help="Aggregation period used for ALINEA measurements.",
    )
    parser.add_argument(
        "--edgedata-period-seconds",
        type=int,
        default=300,
        help="Aggregation period for the mainline edgeData output in seconds.",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=5.0,
        help="Minimum metering rate in percent.",
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        default=95.0,
        help="Maximum metering rate in percent.",
    )
    parser.add_argument(
        "--queue-override-occupancy-threshold",
        type=float,
        default=30.0,
        help=(
            "Queue detector occupancy threshold in percent. Above this value, "
            "the ramp meter is held inactive/open."
        ),
    )
    parser.add_argument(
        "--queue-weight",
        type=float,
        default=1.0,
        help="Objective weight for average normalized queue ratio.",
    )
    parser.add_argument(
        "--occupancy-weight",
        type=float,
        default=5.0,
        help="Objective weight for normalized average occupancy violation.",
    )
    parser.add_argument(
        "--spillback-weight",
        type=float,
        default=10.0,
        help="Objective weight for spillback active share.",
    )
    parser.add_argument(
        "--violation-mode",
        choices=("positive", "absolute"),
        default="positive",
        help=(
            "How occupancy violation is computed. 'positive' clips values below the "
            "target to zero; 'absolute' uses absolute deviation."
        ),
    )
    parser.add_argument(
        "--paired-compare",
        help=(
            "Optional paired comparison between two K_P values, for example "
            "'25,30'. Uses the shared seeds from the grid-search output."
        ),
    )
    parser.add_argument(
        "--save-controller-timeseries-csv",
        action="store_true",
        help=(
            "Write a long-format CSV with controller time series such as "
            "metering_rate, metering_occupancy, queue_length_m, and queue_ratio."
        ),
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Do not run new simulations. Instead, build summary CSVs and plots "
            "from existing output directory/directories."
        ),
    )
    parser.add_argument(
        "--finalize-input-dirs",
        nargs="*",
        help=(
            "Existing calibration output directories to merge in finalize-only mode. "
            "If omitted, the script finalizes the directory given by --output-dir."
        ),
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def build_kp_values(kp_min: float, kp_max: float, kp_step: float) -> list[float]:
    if kp_step <= 0:
        raise ValueError("--kp-step must be greater than zero")
    if kp_max < kp_min:
        raise ValueError("--kp-max must be greater than or equal to --kp-min")

    values: list[float] = []
    current = kp_min
    while current <= kp_max + 1e-9:
        values.append(round(current, 10))
        current += kp_step
    return values


def build_seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        seeds = [int(raw.strip()) for raw in args.seeds.split(",") if raw.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid values were found")
        if len(seeds) != args.replications:
            raise ValueError(
                f"--seeds contains {len(seeds)} value(s), but --replications is "
                f"{args.replications}. Provide exactly one seed per replication."
            )
        return seeds

    rng = random.Random(args.seed_generator_seed)
    upper_bound = 2**31 - 1
    return [rng.randint(1, upper_bound) for _ in range(args.replications)]


def measurement_steps(measurement_period_sec: float, time_step: float) -> int:
    raw_steps = measurement_period_sec / time_step
    rounded_steps = round(raw_steps)
    if not math.isclose(raw_steps, rounded_steps, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            "--measurement-period-sec must be divisible by --time-step so the "
            "controller samples on exact SUMO steps"
        )
    return int(rounded_steps)


def controller_params(args: argparse.Namespace, kp: float) -> dict[str, float | int]:
    return {
        "target_occupancy": args.target_occupancy,
        "K_P": kp,
        "K_I": 0.0,
        "cycle_duration": args.cycle_duration,
        "measurement_period": measurement_steps(
            args.measurement_period_sec,
            args.time_step,
        ),
        "min_rate": args.min_rate,
        "max_rate": args.max_rate,
        "queue_override_occupancy_threshold": args.queue_override_occupancy_threshold,
    }


def create_controllers(params: dict[str, float | int]) -> dict[str, ALINEA]:
    controllers: dict[str, ALINEA] = {}
    for ramp_config in RAMP_CONFIGS:
        ramp_meter = RampMeter(
            tl_id=ramp_config["tl_id"],
            mainline_sensors=ramp_config["mainline_sensors"],
            queue_sensors=ramp_config["queue_sensors"],
            queue_override_sensors=ramp_config["queue_override_sensors"],
        )
        controllers[ramp_config["tl_id"]] = ALINEA(
            params=dict(params),
            ramp_meter=ramp_meter,
        )
    return controllers


def series_values(series: list[list[float]]) -> np.ndarray:
    if len(series) <= 1:
        return np.array([], dtype=float)
    return np.array([row[1] for row in series[1:]], dtype=float)


def compute_occupancy_violation(
    occupancies: np.ndarray,
    target_occupancy: float,
    mode: str,
) -> np.ndarray:
    if mode == "absolute":
        return np.abs(occupancies - target_occupancy)
    return np.clip(occupancies - target_occupancy, 0.0, None)


def parse_tripinfo_metrics(tripinfo_path: Path) -> dict[str, float]:
    durations: list[float] = []
    time_losses: list[float] = []
    waiting_times: list[float] = []
    depart_delays: list[float] = []

    if tripinfo_path.exists():
        for _, element in ET.iterparse(tripinfo_path, events=("end",)):
            if element.tag != "tripinfo":
                continue
            durations.append(float(element.attrib.get("duration", 0.0)))
            time_losses.append(float(element.attrib.get("timeLoss", 0.0)))
            waiting_times.append(float(element.attrib.get("waitingTime", 0.0)))
            depart_delays.append(float(element.attrib.get("departDelay", 0.0)))
            element.clear()

    def safe_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "trip_count": float(len(durations)),
        "mean_trip_duration_s": safe_mean(durations),
        "mean_trip_time_loss_s": safe_mean(time_losses),
        "mean_trip_waiting_time_s": safe_mean(waiting_times),
        "mean_trip_depart_delay_s": safe_mean(depart_delays),
        "sum_trip_time_loss_s": float(np.sum(time_losses)) if time_losses else float("nan"),
    }


def compute_run_metrics(
    controllers: dict[str, ALINEA],
    target_occupancy: float,
    queue_weight: float,
    occupancy_weight: float,
    spillback_weight: float,
    violation_mode: str,
    tripinfo_path: Path,
) -> dict[str, float]:
    queue_values: list[np.ndarray] = []
    queue_ratio_values: list[np.ndarray] = []
    occupancy_values: list[np.ndarray] = []
    spillback_values: list[np.ndarray] = []

    for controller in controllers.values():
        queue_series = series_values(controller.measurement_data["queue_length_m"])
        queue_ratio_series = series_values(controller.measurement_data["queue_ratio"])
        occupancy_series = series_values(controller.measurement_data["metering_occupancy"])
        spillback_series = series_values(
            controller.measurement_data["queue_override_active"]
        )
        if queue_series.size:
            queue_values.append(queue_series)
        if queue_ratio_series.size:
            queue_ratio_values.append(queue_ratio_series)
        if occupancy_series.size:
            occupancy_values.append(occupancy_series)
        if spillback_series.size:
            spillback_values.append(spillback_series)

    queue_array = (
        np.concatenate(queue_values).astype(float)
        if queue_values
        else np.array([], dtype=float)
    )
    queue_ratio_array = (
        np.concatenate(queue_ratio_values).astype(float)
        if queue_ratio_values
        else np.array([], dtype=float)
    )
    occupancy_array = (
        np.concatenate(occupancy_values).astype(float)
        if occupancy_values
        else np.array([], dtype=float)
    )
    spillback_array = (
        np.concatenate(spillback_values).astype(float)
        if spillback_values
        else np.array([], dtype=float)
    )
    violation_array = compute_occupancy_violation(
        occupancy_array,
        target_occupancy,
        violation_mode,
    )

    avg_queue_length = float(np.mean(queue_array)) if queue_array.size else float("nan")
    avg_queue_ratio = (
        float(np.nanmean(queue_ratio_array)) if queue_ratio_array.size else float("nan")
    )
    avg_occupancy_violation = (
        float(np.mean(violation_array)) if violation_array.size else float("nan")
    )
    avg_normalized_occupancy_violation = (
        avg_occupancy_violation / target_occupancy
        if target_occupancy > 0 and math.isfinite(avg_occupancy_violation)
        else float("nan")
    )
    spillback_active_share = (
        float(np.nanmean(spillback_array)) if spillback_array.size else float("nan")
    )
    objective = (
        queue_weight * avg_queue_ratio
        + occupancy_weight * avg_normalized_occupancy_violation
        + spillback_weight * spillback_active_share
    )

    metrics = {
        "objective": objective,
        "avg_queue_ratio": avg_queue_ratio,
        "avg_normalized_occupancy_violation": avg_normalized_occupancy_violation,
        "spillback_active_share": spillback_active_share,
        "avg_queue_length_m": avg_queue_length,
        "avg_occupancy_violation": avg_occupancy_violation,
        "queue_observation_count": float(queue_array.size),
        "occupancy_observation_count": float(occupancy_array.size),
    }
    metrics.update(parse_tripinfo_metrics(tripinfo_path))
    return metrics


def extract_controller_timeseries(
    controllers: dict[str, ALINEA],
) -> dict[str, dict[str, list[list[float]]]]:
    snapshot: dict[str, dict[str, list[list[float]]]] = {}
    for tl_id, controller in controllers.items():
        snapshot[tl_id] = {
            metric_key: [list(row) for row in controller.measurement_data[metric_key]]
            for metric_key, _ in TIMESERIES_METRICS
        }
    return snapshot


def write_tls_state_debug_csv(
    output_path: Path,
    state_rows: list[dict[str, float | str]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["time_sec", "tl_id", "state"],
        )
        writer.writeheader()
        for row in state_rows:
            writer.writerow(row)


def write_controller_timeseries_csv(
    output_path: Path,
    timeseries: dict[str, dict[str, list[list[float]]]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["tl_id", "metric", "time_sec", "value"],
        )
        writer.writeheader()
        for tl_id, ramp_timeseries in timeseries.items():
            for metric_key, _ in TIMESERIES_METRICS:
                for time_sec, value in ramp_timeseries[metric_key]:
                    writer.writerow(
                        {
                            "tl_id": tl_id,
                            "metric": metric_key,
                            "time_sec": time_sec,
                            "value": value,
                        }
                    )


def write_edgedata_additional_file(run_dir: Path, period_seconds: int) -> Path:
    run_slug = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        os.path.relpath(run_dir, start=Path.cwd()),
    )
    additional_path = Path.cwd() / f".alinea_edgedata_{run_slug}.add.xml"
    additional_path.write_text(
        "\n".join(
            [
                "<additional>",
                (
                    "  <edgeData id=\"mainline_edgedata\" file=\"edgedata.xml\" "
                    f"freq=\"{period_seconds}\"/>"
                ),
                "</additional>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return additional_path


def remove_edgedata_additional_file(run_dir: Path) -> None:
    run_slug = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        os.path.relpath(run_dir, start=Path.cwd()),
    )
    try:
        (Path.cwd() / f".alinea_edgedata_{run_slug}.add.xml").unlink()
    except FileNotFoundError:
        pass


def sumo_command(
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
) -> list[str]:
    # SUMO output-prefix handling is more reliable with a path relative to the
    # current working directory than with an absolute path.
    output_prefix = os.path.relpath(run_dir, start=Path.cwd())
    edgedata_additional_file = write_edgedata_additional_file(
        run_dir,
        args.edgedata_period_seconds,
    )
    additional_files = [
        resolve_path("detSIM.add.xml"),
        resolve_path("detRM.add.xml"),
        resolve_path("detDFR.add.xml"),
        edgedata_additional_file,
    ]
    return [
        args.sumo_binary,
        "-c",
        str(resolve_path(args.sumo_config)),
        "--start",
        "--quit-on-end",
        "--no-warnings",
        "--duration-log.statistics",
        "--time-to-teleport",
        "-1",
        "--seed",
        str(seed),
        "--additional-files",
        ",".join(str(path) for path in additional_files),
        "--output-prefix",
        f"{output_prefix}/",
        "--lanechange-output",
        "lanechanges.xml",
        "--tripinfo-output",
        "tripinfo.xml",
        "--device.emissions.probability",
        "1",
        "--log",
        "sumo.log",
    ]


def run_single_replication(
    args: argparse.Namespace,
    kp: float,
    seed: int,
    run_dir: Path,
) -> tuple[dict[str, float], dict[str, dict[str, list[list[float]]]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    controllers = create_controllers(controller_params(args, kp))
    command = sumo_command(args, seed, run_dir)
    started = False
    tls_state_rows: list[dict[str, float | str]] = []
    last_tls_states: dict[str, str] = {}

    try:
        traci.start(command, port=args.traci_port)
        started = True
        for tl_id in controllers:
            state = traci.trafficlight.getRedYellowGreenState(tl_id)
            last_tls_states[tl_id] = state
            tls_state_rows.append(
                {
                    "time_sec": 0.0,
                    "tl_id": tl_id,
                    "state": state,
                }
            )
        while True:
            if (
                args.duration_sec is None
                and traci.simulation.getMinExpectedNumber() <= 0
            ):
                break
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            for tl_id, controller in controllers.items():
                controller.execute_control(current_time)
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                if state != last_tls_states[tl_id]:
                    tls_state_rows.append(
                        {
                            "time_sec": current_time,
                            "tl_id": tl_id,
                            "state": state,
                        }
                    )
                    last_tls_states[tl_id] = state
            if args.duration_sec is not None and current_time >= args.duration_sec:
                break
    finally:
        remove_edgedata_additional_file(run_dir)
        if started:
            try:
                traci.close()
            except Exception:
                pass

    write_tls_state_debug_csv(run_dir / "tls_state_changes.csv", tls_state_rows)

    metrics = compute_run_metrics(
        controllers=controllers,
        target_occupancy=args.target_occupancy,
        queue_weight=args.queue_weight,
        occupancy_weight=args.occupancy_weight,
        spillback_weight=args.spillback_weight,
        violation_mode=args.violation_mode,
        tripinfo_path=run_dir / "tripinfo.xml",
    )
    timeseries = extract_controller_timeseries(controllers)
    if args.save_controller_timeseries_csv:
        write_controller_timeseries_csv(
            run_dir / "controller_timeseries.csv",
            timeseries,
        )
    return metrics, timeseries


def run_replication_task(
    args: argparse.Namespace,
    kp: float,
    replication: int,
    seed: int,
    run_dir: str,
) -> ReplicationResult:
    metrics, timeseries = run_single_replication(
        args=args,
        kp=kp,
        seed=seed,
        run_dir=Path(run_dir),
    )
    return {
        "kp": kp,
        "replication": replication,
        "seed": seed,
        "metrics": metrics,
        "timeseries": timeseries,
    }


def sanitize_value(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.10g}"


def write_seed_csv(output_path: Path, seeds: list[int]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["replication", "seed"])
        writer.writeheader()
        for replication, seed in enumerate(seeds, start=1):
            writer.writerow({"replication": replication, "seed": seed})


def initialize_raw_results_csv(output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_RESULTS_FIELDNAMES)
        writer.writeheader()


def append_raw_result_row(
    output_path: Path,
    row: dict[str, float | int],
) -> None:
    with output_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_RESULTS_FIELDNAMES)
        writer.writerow(row)


def write_raw_results_csv(output_path: Path, rows: list[dict[str, float | int]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_RESULTS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_raw_results_csv(output_path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if not output_path.exists():
        return rows

    int_fields = {"replication", "seed"}
    with output_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for csv_row in reader:
            parsed_row: dict[str, float | int] = {}
            missing_required = False
            for field in RAW_RESULTS_FIELDNAMES:
                raw_value = csv_row.get(field)
                if raw_value is None or raw_value == "":
                    missing_required = True
                    break
                if field in int_fields:
                    parsed_row[field] = int(raw_value)
                else:
                    parsed_row[field] = float(raw_value)
            if not missing_required:
                rows.append(parsed_row)
    return rows


def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    return 1.96


def summarize_series(values: np.ndarray) -> dict[str, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "stddev": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    mean_value = float(np.mean(clean))
    if clean.size >= 2:
        stddev_value = float(np.std(clean, ddof=1))
        margin = t_critical_95(clean.size - 1) * stddev_value / math.sqrt(clean.size)
    else:
        stddev_value = float("nan")
        margin = float("nan")

    return {
        "n": float(clean.size),
        "mean": mean_value,
        "stddev": stddev_value,
        "median": float(np.median(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "ci95_low": mean_value - margin if math.isfinite(margin) else float("nan"),
        "ci95_high": mean_value + margin if math.isfinite(margin) else float("nan"),
    }


def summarize_results(rows: list[dict[str, float | int]]) -> list[dict[str, float]]:
    grouped: dict[float, list[dict[str, float | int]]] = {}
    for row in rows:
        grouped.setdefault(float(row["kp"]), []).append(row)

    summary_rows: list[dict[str, float]] = []
    for kp in sorted(grouped):
        group = grouped[kp]
        summary_row: dict[str, float] = {"kp": kp, "replications": float(len(group))}
        for metric in SUMMARY_METRICS:
            series = np.array([float(item[metric]) for item in group], dtype=float)
            stats = summarize_series(series)
            for stat_name, stat_value in stats.items():
                summary_row[f"{metric}_{stat_name}"] = stat_value
        summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: row["kp"])
    return summary_rows


def write_summary_csv(output_path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["kp", "replications"]
    for metric in SUMMARY_METRICS:
        fieldnames.extend(
            [
                f"{metric}_n",
                f"{metric}_mean",
                f"{metric}_stddev",
                f"{metric}_median",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_ci95_low",
                f"{metric}_ci95_high",
            ]
        )

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: sanitize_value(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def write_ranking_csv(output_path: Path, rows: list[dict[str, float]]) -> None:
    ranked_rows = sorted(rows, key=lambda row: row["objective_mean"])
    write_summary_csv(output_path, ranked_rows)


def plot_metric_profiles(output_path: Path, rows: list[dict[str, float]]) -> None:
    metrics = [
        ("objective", "Objective", "#1d4ed8"),
        ("avg_queue_ratio", "Avg queue ratio", "#d97706"),
        (
            "avg_normalized_occupancy_violation",
            "Normalized occupancy violation",
            "#0f766e",
        ),
        ("spillback_active_share", "Spillback active share", "#be123c"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5))
    if len(metrics) == 1:
        axes = [axes]

    kp_values = np.array([row["kp"] for row in rows], dtype=float)
    for ax, (metric_key, title, color) in zip(axes, metrics):
        means = np.array([row[f"{metric_key}_mean"] for row in rows], dtype=float)
        ci_low = np.array([row[f"{metric_key}_ci95_low"] for row in rows], dtype=float)
        ci_high = np.array([row[f"{metric_key}_ci95_high"] for row in rows], dtype=float)
        yerr = np.vstack((means - ci_low, ci_high - means))
        invalid_mask = ~np.isfinite(yerr)
        yerr[invalid_mask] = 0.0

        ax.errorbar(
            kp_values,
            means,
            yerr=yerr,
            fmt="-o",
            color=color,
            ecolor="#334155",
            elinewidth=1.3,
            capsize=4,
            linewidth=2.0,
            markersize=6,
        )
        ax.set_title(title, fontsize=17, pad=12)
        ax.set_xlabel("K_P", fontsize=15)
        ax.set_ylabel("Mean with 95% CI", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(color="#d1d5db", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_objective_boxplot(
    output_path: Path,
    rows: list[dict[str, float | int]],
    kp_values: list[float],
) -> None:
    grouped = {
        kp: np.array(
            [
                float(row["objective"])
                for row in rows
                if math.isclose(float(row["kp"]), kp, rel_tol=0, abs_tol=1e-9)
            ],
            dtype=float,
        )
        for kp in kp_values
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(
        [grouped[kp] for kp in kp_values],
        tick_labels=[f"{kp:g}" for kp in kp_values],
        patch_artist=True,
        boxprops={"facecolor": "#dbeafe", "edgecolor": "#1d4ed8"},
        medianprops={"color": "#111827", "linewidth": 1.5},
        whiskerprops={"color": "#1d4ed8"},
        capprops={"color": "#1d4ed8"},
    )
    ax.set_title("ALINEA objective distribution by K_P")
    ax.set_xlabel("K_P")
    ax.set_ylabel("Objective")
    ax.grid(color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_objective_spaghetti(
    output_path: Path,
    rows: list[dict[str, float | int]],
    kp_values: list[float],
) -> None:
    values_by_seed: dict[int, dict[float, float]] = {}
    values_by_kp: dict[float, list[float]] = {kp: [] for kp in kp_values}
    for row in rows:
        kp = float(row["kp"])
        seed = int(row["seed"])
        objective = float(row["objective"])
        values_by_seed.setdefault(seed, {})[kp] = objective
        if kp in values_by_kp:
            values_by_kp[kp].append(objective)

    fig, ax = plt.subplots(figsize=(12, 6))
    for seed, seed_values in sorted(values_by_seed.items()):
        xs = [kp for kp in kp_values if kp in seed_values]
        if len(xs) < 2:
            continue
        ys = [seed_values[kp] for kp in xs]
        ax.plot(
            xs,
            ys,
            color="#64748b",
            linewidth=1.0,
            alpha=0.35,
            marker="o",
            markersize=3.5,
        )

    mean_values = [
        float(np.mean(values_by_kp[kp])) if values_by_kp[kp] else np.nan
        for kp in kp_values
    ]
    ax.plot(
        kp_values,
        mean_values,
        color="#be123c",
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        label="Mean objective",
    )
    ax.set_title("ALINEA paired objective by K_P")
    ax.set_xlabel("K_P")
    ax.set_ylabel("Objective")
    ax.set_xticks(kp_values)
    ax.set_xticklabels([f"{kp:g}" for kp in kp_values])
    ax.grid(color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def kp_output_slug(kp: float) -> str:
    return f"{kp:g}".replace(".", "p")


def parse_run_identifiers(run_dir: Path) -> tuple[float, int, int] | None:
    kp_dir = run_dir.parent
    if not kp_dir.name.startswith("kp_"):
        return None
    run_match = RUN_DIR_PATTERN.fullmatch(run_dir.name)
    if run_match is None:
        return None
    try:
        kp = float(kp_dir.name.removeprefix("kp_").replace("p", "."))
    except ValueError:
        return None
    replication = int(run_match.group(1))
    seed = int(run_match.group(2))
    return kp, replication, seed


def load_controller_timeseries_csv(
    output_path: Path,
) -> dict[str, dict[str, list[list[float]]]]:
    timeseries = {
        ramp_config["tl_id"]: {
            metric_key: []
            for metric_key, _ in TIMESERIES_METRICS
        }
        for ramp_config in RAMP_CONFIGS
    }
    if not output_path.exists():
        return timeseries

    with output_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tl_id = row.get("tl_id")
            metric = row.get("metric")
            time_sec = row.get("time_sec")
            value = row.get("value")
            if (
                tl_id is None
                or metric is None
                or time_sec is None
                or value is None
                or tl_id not in timeseries
                or metric not in timeseries[tl_id]
            ):
                continue
            timeseries[tl_id][metric].append([float(time_sec), float(value)])
    return timeseries


def load_run_plot_rows_from_output_dirs(output_dirs: list[Path]) -> list[RunPlotRow]:
    run_plot_rows: list[RunPlotRow] = []
    for output_dir in output_dirs:
        for csv_path in sorted(output_dir.glob("runs/kp_*/rep_*/controller_timeseries.csv")):
            identifiers = parse_run_identifiers(csv_path.parent)
            if identifiers is None:
                continue
            kp, replication, seed = identifiers
            run_plot_rows.append(
                {
                    "kp": kp,
                    "replication": replication,
                    "seed": seed,
                    "timeseries": load_controller_timeseries_csv(csv_path),
                }
            )
    return run_plot_rows


def plot_kp_controller_timeseries(
    output_path: Path,
    kp: float,
    runs: list[RunPlotRow],
    ramp_configs: list[RampConfig],
    direction_label: str,
) -> None:
    if not runs:
        return

    ncols = len(runs)
    fig, axes = plt.subplots(
        len(TIMESERIES_METRICS),
        ncols,
        figsize=(4.0 * ncols, 2.7 * len(TIMESERIES_METRICS)),
        squeeze=False,
    )
    legend_handles = None
    legend_labels = None

    for col_idx, run in enumerate(runs):
        timeseries = run["timeseries"]
        for row_idx, (metric_key, ylabel) in enumerate(TIMESERIES_METRICS):
            ax = axes[row_idx][col_idx]
            for ramp_idx, ramp_config in enumerate(ramp_configs):
                tl_id = ramp_config["tl_id"]
                series = np.asarray(timeseries[tl_id][metric_key], dtype=float)
                if series.ndim != 2 or series.shape[0] == 0:
                    continue
                style = controller_line_style(ramp_idx)
                ax.step(
                    series[:, 0],
                    series[:, 1],
                    where="post",
                    linewidth=1.2,
                    alpha=0.8,
                    label=tl_id,
                    **style,
                )
            ax.set_title(
                f"rep {int(run['replication'])}\nseed {int(run['seed'])}",
                fontsize=CONTROLLER_SUBPLOT_TITLE_FONTSIZE,
            )
            ax.grid(color="#d1d5db", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=CONTROLLER_AXIS_LABEL_FONTSIZE)
            if row_idx == len(TIMESERIES_METRICS) - 1:
                ax.set_xlabel(
                    "Simulation time [s]",
                    fontsize=CONTROLLER_AXIS_LABEL_FONTSIZE,
                )
            ax.tick_params(axis="both", labelsize=CONTROLLER_TICK_FONTSIZE)
            if legend_handles is None and row_idx == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 5),
            frameon=False,
            fontsize=CONTROLLER_LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def median_timeseries(
    runs: list[RunPlotRow],
    tl_id: str,
    metric_key: str,
) -> np.ndarray:
    values_by_time: dict[float, list[float]] = {}
    for run in runs:
        series = np.asarray(run["timeseries"][tl_id][metric_key], dtype=float)
        if series.ndim != 2 or series.shape[0] == 0 or series.shape[1] < 2:
            continue
        for time_sec, value in series[:, :2]:
            values_by_time.setdefault(float(time_sec), []).append(float(value))

    rows = []
    for time_sec in sorted(values_by_time):
        values = np.asarray(values_by_time[time_sec], dtype=float)
        if values.size == 0 or np.all(~np.isfinite(values)):
            continue
        rows.append([time_sec, float(np.nanmedian(values))])
    return np.asarray(rows, dtype=float)


def rolling_median_timeseries(
    series: np.ndarray,
    window_sec: float,
) -> np.ndarray:
    if series.ndim != 2 or series.shape[0] == 0 or series.shape[1] < 2:
        return series
    if window_sec <= 0:
        return series

    times = series[:, 0]
    values = series[:, 1]
    half_window = window_sec / 2.0
    smoothed_values = np.empty_like(values, dtype=float)

    for idx, time_sec in enumerate(times):
        left = int(np.searchsorted(times, time_sec - half_window, side="left"))
        right = int(np.searchsorted(times, time_sec + half_window, side="right"))
        smoothed_values[idx] = float(np.nanmedian(values[left:right]))

    return np.column_stack((times, smoothed_values))


def plot_kp_controller_median_timeseries(
    output_path: Path,
    kp: float,
    runs: list[RunPlotRow],
    ramp_configs: list[RampConfig],
    direction_label: str,
    metric_key: str,
    ylabel: str,
) -> None:
    if not runs:
        return

    fig, ax = plt.subplots(figsize=CONTROLLER_MEDIAN_FIGSIZE)
    plotted = False

    for ramp_idx, ramp_config in enumerate(ramp_configs):
        tl_id = str(ramp_config["tl_id"])
        series = rolling_median_timeseries(
            median_timeseries(runs, tl_id, metric_key),
            CONTROLLER_ROLLING_MEDIAN_WINDOW_SEC,
        )
        if series.ndim != 2 or series.shape[0] == 0:
            continue
        plotted = True
        style = controller_line_style(ramp_idx)
        ax.plot(
            series[:, 0],
            series[:, 1],
            linewidth=2.4,
            alpha=0.95,
            label=tl_id,
            **style,
        )

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Simulation time [s]", fontsize=CONTROLLER_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=CONTROLLER_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=CONTROLLER_TICK_FONTSIZE)
    ax.grid(color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 2),
            frameon=False,
            fontsize=CONTROLLER_MEDIAN_LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, 0.99),
        )
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.76)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def paired_sign_test_pvalue(diff: np.ndarray) -> float:
    clean = diff[np.isfinite(diff)]
    clean = clean[clean != 0]
    n = int(clean.size)
    if n == 0:
        return float("nan")

    positive = int(np.sum(clean > 0))
    negative = n - positive
    smaller_tail = min(positive, negative)
    cumulative = 0.0
    for k in range(smaller_tail + 1):
        cumulative += math.comb(n, k) / (2**n)
    return min(1.0, 2.0 * cumulative)


def paired_objective_differences(
    rows: list[dict[str, float | int]],
    kp_a: float,
    kp_b: float,
) -> tuple[np.ndarray, list[int]]:
    rows_a = {
        int(row["seed"]): row
        for row in rows
        if math.isclose(float(row["kp"]), kp_a, rel_tol=0, abs_tol=1e-9)
    }
    rows_b = {
        int(row["seed"]): row
        for row in rows
        if math.isclose(float(row["kp"]), kp_b, rel_tol=0, abs_tol=1e-9)
    }
    shared_seeds = sorted(set(rows_a) & set(rows_b))
    if not shared_seeds:
        raise ValueError(f"No paired seeds available for K_P {kp_a:g} and {kp_b:g}")

    objective_diff = np.array(
        [
            float(rows_b[seed]["objective"]) - float(rows_a[seed]["objective"])
            for seed in shared_seeds
        ],
        dtype=float,
    )
    return objective_diff, shared_seeds


def paired_compare_rows(
    rows: list[dict[str, float | int]],
    kp_a: float,
    kp_b: float,
) -> dict[str, float]:
    objective_diff, _shared_seeds = paired_objective_differences(rows, kp_a, kp_b)
    clean = objective_diff[np.isfinite(objective_diff)]
    stats = summarize_series(clean)
    stddev = stats["stddev"]
    if clean.size >= 2 and math.isfinite(stddev) and stddev > 0:
        t_stat = stats["mean"] / (stddev / math.sqrt(clean.size))
        effect_size_dz = stats["mean"] / stddev
    else:
        t_stat = float("nan")
        effect_size_dz = float("nan")

    return {
        "kp_a": kp_a,
        "kp_b": kp_b,
        "paired_replications": float(clean.size),
        "objective_diff_mean_b_minus_a": stats["mean"],
        "objective_diff_stddev": stats["stddev"],
        "objective_diff_ci95_low": stats["ci95_low"],
        "objective_diff_ci95_high": stats["ci95_high"],
        "paired_t_statistic": t_stat,
        "paired_sign_test_pvalue": paired_sign_test_pvalue(clean),
        "cohens_dz": effect_size_dz,
    }


def plot_paired_difference_histogram(
    output_path: Path,
    objective_diff: np.ndarray,
    kp_a: float,
    kp_b: float,
) -> None:
    clean = objective_diff[np.isfinite(objective_diff)]
    if clean.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = max(4, min(12, int(math.ceil(math.sqrt(clean.size)))))
    ax.hist(clean, bins=bins, color="#dbeafe", edgecolor="#1d4ed8", linewidth=1.1)
    ax.axvline(0, color="#111827", linestyle="--", linewidth=1.2, label="No difference")
    ax.axvline(
        float(np.mean(clean)),
        color="#be123c",
        linewidth=1.8,
        label="Mean difference",
    )
    ax.set_title(
        f"Paired objective differences: K_P={kp_b:g} minus K_P={kp_a:g}"
    )
    ax.set_xlabel("Objective difference")
    ax.set_ylabel("Seed count")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paired_difference_qq(
    output_path: Path,
    objective_diff: np.ndarray,
    kp_a: float,
    kp_b: float,
) -> None:
    clean = np.sort(objective_diff[np.isfinite(objective_diff)])
    fig, ax = plt.subplots(figsize=(6, 6))
    if clean.size >= 2:
        mean_value = float(np.mean(clean))
        stddev_value = float(np.std(clean, ddof=1))
        if math.isfinite(stddev_value) and stddev_value > 0:
            normal = NormalDist()
            theoretical = np.array(
                [normal.inv_cdf((idx - 0.5) / clean.size) for idx in range(1, clean.size + 1)],
                dtype=float,
            )
            observed = (clean - mean_value) / stddev_value
            ax.scatter(theoretical, observed, color="#1d4ed8", s=36)
            axis_min = float(min(np.min(theoretical), np.min(observed)))
            axis_max = float(max(np.max(theoretical), np.max(observed)))
            ax.plot(
                [axis_min, axis_max],
                [axis_min, axis_max],
                color="#be123c",
                linewidth=1.5,
            )
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
        else:
            ax.text(
                0.5,
                0.5,
                "All finite differences are identical.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "At least two finite paired differences are required.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(f"Q-Q plot of paired differences: K_P={kp_b:g} minus K_P={kp_a:g}")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Observed standardized differences")
    ax.grid(color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_paired_comparison_csv(output_path: Path, row: dict[str, float]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow({key: sanitize_value(value) for key, value in row.items()})


def capture_sumo_version(sumo_binary: str) -> str:
    try:
        result = subprocess.run(
            [sumo_binary, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""

    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else ""


def write_metadata(
    output_path: Path,
    args: argparse.Namespace,
    kp_values: list[float],
    seeds: list[int],
) -> None:
    metadata = {
        "sumo_config": str(resolve_path(args.sumo_config)),
        "sumo_binary": args.sumo_binary,
        "traci_port": args.traci_port,
        "sumo_version": capture_sumo_version(args.sumo_binary),
        "duration_sec": args.duration_sec,
        "time_step": args.time_step,
        "measurement_period_sec": args.measurement_period_sec,
        "edgedata_period_seconds": args.edgedata_period_seconds,
        "measurement_period_steps": measurement_steps(
            args.measurement_period_sec,
            args.time_step,
        ),
        "kp_values": kp_values,
        "target_occupancy": args.target_occupancy,
        "cycle_duration": args.cycle_duration,
        "min_rate": args.min_rate,
        "max_rate": args.max_rate,
        "queue_override_occupancy_threshold": args.queue_override_occupancy_threshold,
        "queue_weight": args.queue_weight,
        "occupancy_weight": args.occupancy_weight,
        "spillback_weight": args.spillback_weight,
        "violation_mode": args.violation_mode,
        "replications": len(seeds),
        "parallel_jobs": args.parallel_jobs,
        "seed_generator_seed": args.seed_generator_seed,
        "seeds": seeds,
        "ramp_count": len(RAMP_CONFIGS),
        "ramps": RAMP_CONFIGS,
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def finalize_outputs(
    output_dir: Path,
    timeseries_plot_dir: Path,
    kp_values: list[float],
    raw_rows: list[dict[str, float | int]],
    run_plot_rows: list[RunPlotRow],
    paired_compare: str | None,
) -> None:
    raw_results_path = output_dir / "alinea_stochastic_raw_results.csv"
    summary_path = output_dir / "alinea_stochastic_summary.csv"
    ranking_path = output_dir / "alinea_stochastic_ranking.csv"
    profile_plot_path = output_dir / "alinea_stochastic_profiles.png"
    boxplot_path = output_dir / "alinea_stochastic_objective_boxplot.png"
    spaghetti_plot_path = output_dir / "alinea_stochastic_objective_spaghetti.png"
    paired_histogram_path = output_dir / "alinea_stochastic_best_kp_diff_histogram.png"
    paired_qq_path = output_dir / "alinea_stochastic_best_kp_diff_qq.png"

    if not raw_rows:
        print("No completed replications available; skipped summary and plot generation.")
        return

    unique_raw_rows: dict[tuple[float, int, int], dict[str, float | int]] = {}
    for row in raw_rows:
        key = (float(row["kp"]), int(row["replication"]), int(row["seed"]))
        unique_raw_rows[key] = row
    raw_rows = sorted(
        unique_raw_rows.values(),
        key=lambda row: (float(row["kp"]), int(row["replication"]), int(row["seed"])),
    )
    unique_run_plot_rows: dict[tuple[float, int, int], RunPlotRow] = {}
    for row in run_plot_rows:
        key = (row["kp"], row["replication"], row["seed"])
        unique_run_plot_rows[key] = row
    run_plot_rows = list(unique_run_plot_rows.values())
    write_raw_results_csv(raw_results_path, raw_rows)
    summary_rows = summarize_results(raw_rows)
    write_summary_csv(summary_path, summary_rows)
    write_ranking_csv(ranking_path, summary_rows)
    ranked_summary_rows = sorted(summary_rows, key=lambda row: row["objective_mean"])
    plot_metric_profiles(profile_plot_path, summary_rows)
    plotted_kp_values = sorted({float(row["kp"]) for row in raw_rows})
    plot_objective_boxplot(boxplot_path, raw_rows, plotted_kp_values)
    plot_objective_spaghetti(spaghetti_plot_path, raw_rows, plotted_kp_values)
    if len(ranked_summary_rows) >= 2:
        best_kp = float(ranked_summary_rows[0]["kp"])
        second_best_kp = float(ranked_summary_rows[1]["kp"])
        best_diff, _shared_seeds = paired_objective_differences(
            raw_rows,
            best_kp,
            second_best_kp,
        )
        plot_paired_difference_histogram(
            paired_histogram_path,
            best_diff,
            best_kp,
            second_best_kp,
        )
        plot_paired_difference_qq(
            paired_qq_path,
            best_diff,
            best_kp,
            second_best_kp,
        )
    ramp_configs_by_direction = {
        "B": [config for config in RAMP_CONFIGS if str(config["tl_id"]).endswith("_B")],
        "Z": [config for config in RAMP_CONFIGS if str(config["tl_id"]).endswith("_Z")],
    }
    for kp in plotted_kp_values:
        kp_runs = [
            row
            for row in run_plot_rows
            if math.isclose(row["kp"], kp, rel_tol=0, abs_tol=1e-9)
        ]
        kp_runs.sort(key=lambda row: row["replication"])
        for direction_label, ramp_configs in ramp_configs_by_direction.items():
            plot_kp_controller_timeseries(
                timeseries_plot_dir
                / f"kp_{kp_output_slug(kp)}_controller_timeseries_{direction_label}.png",
                kp,
                kp_runs,
                ramp_configs,
                f"RMS {direction_label}",
            )
            for metric_key, ylabel in TIMESERIES_METRICS:
                plot_kp_controller_median_timeseries(
                    timeseries_plot_dir
                    / (
                        f"kp_{kp_output_slug(kp)}_median_{metric_key}_"
                        f"{direction_label}.png"
                    ),
                    kp,
                    kp_runs,
                    ramp_configs,
                    f"RMS {direction_label}",
                    metric_key,
                    ylabel,
                )

    if paired_compare:
        compare_values = [float(value.strip()) for value in paired_compare.split(",")]
        if len(compare_values) != 2:
            raise ValueError("--paired-compare must contain exactly two K_P values")
        comparison_row = paired_compare_rows(raw_rows, compare_values[0], compare_values[1])
        comparison_path = output_dir / "alinea_stochastic_paired_comparison.csv"
        write_paired_comparison_csv(comparison_path, comparison_row)
        print(f"Wrote paired comparison to {comparison_path}")

    best_row = min(summary_rows, key=lambda row: row["objective_mean"])
    print(f"Wrote raw stochastic results to {raw_results_path}")
    print(f"Wrote summary statistics to {summary_path}")
    print(f"Wrote ranked summary to {ranking_path}")
    print(f"Wrote profile plot to {profile_plot_path}")
    print(f"Wrote objective boxplot to {boxplot_path}")
    print(f"Wrote paired objective spaghetti plot to {spaghetti_plot_path}")
    if len(ranked_summary_rows) >= 2:
        print(f"Wrote best-K_P paired difference histogram to {paired_histogram_path}")
        print(f"Wrote best-K_P paired difference Q-Q plot to {paired_qq_path}")
    print(f"Wrote controller time-series plots to {timeseries_plot_dir}")
    print(
        f"Best K_P by mean objective: {best_row['kp']:.4g} "
        f"(mean={best_row['objective_mean']:.4f}, "
        f"95% CI=[{best_row['objective_ci95_low']:.4f}, "
        f"{best_row['objective_ci95_high']:.4f}])"
    )


def main() -> int:
    args = parse_args()
    if args.parallel_jobs <= 0:
        raise ValueError("--parallel-jobs must be greater than zero")
    if args.parallel_jobs > 1 and args.traci_port is not None:
        raise ValueError("--traci-port cannot be used together with --parallel-jobs > 1")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timeseries_plot_dir = output_dir / "controller_timeseries"
    timeseries_plot_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = output_dir / "alinea_stochastic_raw_results.csv"

    if args.finalize_only:
        input_dirs_arg = args.finalize_input_dirs or [str(output_dir)]
        input_dirs = [resolve_path(path_str) for path_str in input_dirs_arg]
        raw_rows: list[dict[str, float | int]] = []
        for input_dir in input_dirs:
            raw_rows.extend(load_raw_results_csv(input_dir / "alinea_stochastic_raw_results.csv"))
        run_plot_rows = load_run_plot_rows_from_output_dirs(input_dirs)
        finalize_outputs(
            output_dir=output_dir,
            timeseries_plot_dir=timeseries_plot_dir,
            kp_values=[],
            raw_rows=raw_rows,
            run_plot_rows=run_plot_rows,
            paired_compare=args.paired_compare,
        )
        print(f"Finalized existing calibration outputs into {output_dir}")
        return 0

    kp_values = build_kp_values(args.kp_min, args.kp_max, args.kp_step)
    seeds = build_seed_list(args)

    raw_rows: list[dict[str, float | int]] = []
    run_plot_rows: list[RunPlotRow] = []

    write_seed_csv(output_dir / "seed_list.csv", seeds)
    write_metadata(output_dir / "experiment_metadata.json", args, kp_values, seeds)
    initialize_raw_results_csv(raw_results_path)
    interrupted = False

    try:
        for kp in kp_values:
            worker_count = min(args.parallel_jobs, len(seeds))
            if worker_count == 1:
                for replication, seed in enumerate(seeds, start=1):
                    run_dir = runs_dir / f"kp_{kp:g}" / f"rep_{replication:02d}_seed_{seed}"
                    metrics, timeseries = run_single_replication(args, kp, seed, run_dir)
                    raw_row: dict[str, float | int] = {
                        "kp": kp,
                        "replication": replication,
                        "seed": seed,
                        **metrics,
                    }
                    raw_rows.append(raw_row)
                    append_raw_result_row(raw_results_path, raw_row)
                    run_plot_rows.append(
                        {
                            "kp": kp,
                            "replication": replication,
                            "seed": seed,
                            "timeseries": timeseries,
                        }
                    )
                    print(
                        f"Finished K_P={kp:g} replication {replication}/{len(seeds)} "
                        f"(seed={seed}) objective={metrics['objective']:.4f}"
                    )
            else:
                executor = ProcessPoolExecutor(max_workers=worker_count)
                futures: dict[Future[ReplicationResult], tuple[int, int]] = {}
                try:
                    for replication, seed in enumerate(seeds, start=1):
                        run_dir = runs_dir / f"kp_{kp:g}" / f"rep_{replication:02d}_seed_{seed}"
                        future = executor.submit(
                            run_replication_task,
                            args,
                            kp,
                            replication,
                            seed,
                            str(run_dir),
                        )
                        futures[future] = (replication, seed)

                    pending: set[Future[ReplicationResult]] = set(futures)
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            result = future.result()
                            metrics = result["metrics"]
                            raw_row = {
                                "kp": result["kp"],
                                "replication": result["replication"],
                                "seed": result["seed"],
                                **metrics,
                            }
                            raw_rows.append(raw_row)
                            append_raw_result_row(raw_results_path, raw_row)
                            run_plot_rows.append(
                                {
                                    "kp": result["kp"],
                                    "replication": result["replication"],
                                    "seed": result["seed"],
                                    "timeseries": result["timeseries"],
                                }
                            )
                            print(
                                f"Finished K_P={kp:g} replication {result['replication']}/{len(seeds)} "
                                f"(seed={result['seed']}) objective={metrics['objective']:.4f}"
                            )
                except KeyboardInterrupt:
                    interrupted = True
                    print(
                        "\nKeyboardInterrupt received. Stopping pending replications "
                        f"for K_P={kp:g} and finalizing completed results..."
                    )
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                finally:
                    if not interrupted:
                        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Finalizing outputs from completed replications...")

    finalize_outputs(
        output_dir=output_dir,
        timeseries_plot_dir=timeseries_plot_dir,
        kp_values=kp_values,
        raw_rows=raw_rows,
        run_plot_rows=run_plot_rows,
        paired_compare=args.paired_compare,
    )

    print(f"Wrote seed list to {output_dir / 'seed_list.csv'}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
