from __future__ import annotations

import statistics
import math
import re
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    samples = []
    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        samples.append((bench.clock() - start) / 1_000_000.0)
    return samples

def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    source = "leading prefix above (1 + 0.5) x median of the run's second half"
    if len(samples) < 4:
        return unknown(source, "too few samples to establish a settled rate")
    settled = statistics.median(samples[len(samples) // 2:])
    if settled <= 0:
        return unknown(source, "settled median must be positive")
    threshold = settled * (1 + WARMUP_TOL)
    boundary = 0
    for sample in samples:
        if sample <= threshold:
            break
        boundary += 1
    return measured(boundary, source, settled_rate_ms=round(settled, 4),
                    threshold_ms=round(threshold, 4), tolerance=WARMUP_TOL,
                    retained=len(samples) - boundary)

def summarize(samples: list[float]) -> dict[str, Any]:
    keys = ("mean", "std", "min", "max", "p50", "p95", "p99")
    if not samples:
        return {"n": 0, **dict.fromkeys(keys)}
    ordered = sorted(samples)
    n = len(ordered)
    def percentile(q: int) -> float:
        h = (n - 1) * q / 100
        i = math.floor(h)
        return ordered[i] + (h - i) * (ordered[min(i + 1, n - 1)] - ordered[i])
    return {"n": n, "mean": round(statistics.fmean(samples), 4),
            "std": round(statistics.stdev(samples), 4) if n > 2 else 0.0,
            "min": round(ordered[0], 4), "max": round(ordered[-1], 4),
            **{f"p{q}": round(percentile(q), 4) for q in PERCENTILES}}

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    source = ("widest trimmed gap >= 20.0x the median gap, with >= 10% "
              "of samples on each side")
    if len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown(source, "not enough samples to assess modality")
    ordered = sorted(samples)
    trim = int(len(ordered) * 0.05)
    ordered = ordered[trim:len(ordered) - trim]
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    typical = statistics.median(gaps)
    if typical <= 0:
        return unknown(source, "timer resolution is too coarse to resolve gaps")
    widest = max(gaps)
    split = gaps.index(widest) + 1
    # Use the trimmed values to locate the gap, then describe the full run's
    # populations so subgroup counts account for every recorded iteration.
    left = sorted(sample for sample in samples if sample <= ordered[split - 1])
    right = sorted(sample for sample in samples if sample > ordered[split - 1])
    ratio = widest / typical
    modes = [{"n": len(group), "share": round(len(group) / len(samples), 4),
              "median_ms": round(statistics.median(group), 4)}
             for group in (left, right)]
    return measured(ratio >= MULTIMODAL_GAP_RATIO and
                    all(len(group) / len(samples) >= MIN_MODE_FRACTION
                        for group in (left, right)), source,
                    gap_ratio=round(ratio, 2), widest_gap_ms=round(widest, 3),
                    typical_gap_ms=round(typical, 5), modes=modes)

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================

def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel", "-q"])
    if not result.ok or result.returncode != 0:
        return unknown(result.source, result.error or
                       f"command exited with status {result.returncode}")
    lines = result.stdout.splitlines()
    name = None
    index = None
    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            name = line.split("NV Power Mode:", 1)[1].strip()
            if i + 1 < len(lines):
                match = re.search(r"\d+", lines[i + 1])
                if match:
                    index = int(match.group())
            break
    if not name or index is None:
        return unknown(result.source, "could not parse power mode name and index")
    minimum = read_text(bench.telemetry, CPUFREQ_MIN)
    maximum = read_text(bench.telemetry, CPUFREQ_MAX)
    clock_source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"
    try:
        min_hz = int(minimum) if minimum is not None else None
        max_hz = int(maximum) if maximum is not None else None
    except ValueError:
        min_hz = max_hz = None
    if min_hz is None or max_hz is None:
        clocks = None
        finding = unknown(clock_source, "CPU frequency limits could not be read")
    else:
        clocks = min_hz == max_hz
        finding = measured(f"scaling_min_freq={min_hz}, scaling_max_freq={max_hz}",
                           clock_source)
    return measured(name, result.source, mode_index=index,
                    jetson_clocks=clocks, jetson_clocks_source=finding)

def probe_telemetry(bench: Bench) -> dict[str, Any]:
    zones = []
    thermal_dir = bench.telemetry / THERMAL_ZONES
    try:
        paths = sorted(thermal_dir.glob("thermal_zone*/temp"))
    except OSError:
        paths = []
    for path in paths:
        try:
            raw = int(path.read_text().strip())
        except (OSError, ValueError, TypeError):
            continue
        if raw <= -1000:
            continue
        zone = read_text(bench.telemetry,
                         str(path.parent.relative_to(bench.telemetry) / "type"))
        zones.append((raw / 1000.0, zone or path.parent.name))
    if zones:
        hottest = max(zones, key=lambda item: item[0])
        temperature = measured(round(hottest[0], 2), f"{THERMAL_ZONES}/*/temp",
                               zone=hottest[1], zones_read=len(zones))
    else:
        temperature = unknown(f"{THERMAL_ZONES}/*/temp", "no valid thermal zones found")
    power_read = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    power_source = " | ".join(POWER_RAIL_CANDIDATES)
    try:
        power = measured(int(power_read[1]), power_read[0]) if power_read else unknown(
            power_source, "none of the documented INA3221 rail paths could be read")
    except ValueError:
        power = unknown(power_read[0], "power reading is not an integer")
    load_read = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    try:
        gpu = measured(round(int(load_read[1]) / 10.0, 2), load_read[0],
                       units="per-mille / 10") if load_read else unknown(
                           " | ".join(GPU_LOAD_CANDIDATES), "no GPU load file could be read")
    except ValueError:
        gpu = unknown(load_read[0], "GPU load reading is not an integer")
    return {"temperature_c": temperature, "power_mw": power,
            "gpu_utilization_percent": gpu}

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

