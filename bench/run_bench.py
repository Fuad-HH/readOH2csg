"""Scaling and regression benchmark for the native executables.

Runs ``convert2openmc --timing`` on synthetic meshes (bench/make_mesh.cpp) and on
the tagged meshes listed in meshes.toml (``convert2degas2``), five times after a
warm-up, and writes per-stage medians, peak RSS and cross-space copy counts as
JSON. With ``--baseline`` it fails when any stage is >15% slower.

    python bench/run_bench.py --build build/dev --out bench/_results/run.json
"""

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
REPEAT, TOLERANCE = 5, 1.15


def run(cmd, env):
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    stages = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "s":
            stages[parts[0]] = float(parts[1])
    return stages


def bench(name, cmd, counter):
    env = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        copies = Path(tmp) / "copies.json"
        if counter:
            env |= {
                "KOKKOS_TOOLS_LIBS": str(counter),
                "OMEGAH2CSG_COPY_COUNTER_OUT": str(copies),
            }
        out = Path(tmp) / "out"
        runs = [
            run([*cmd, "-o", str(out), "--force", "--timing"], env)
            for _ in range(REPEAT + 1)
        ][1:]
        result = {s: statistics.median(r[s] for r in runs) for s in runs[0]}
        result["total"] = sum(result.values())
        result["peak_rss_mb"] = (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
        )
        if copies.is_file():
            result |= json.loads(copies.read_text())
    print(name, json.dumps(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--build", type=Path, required=True, help="CMake build dir (bench enabled)"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--max-size", type=int, default=SIZES[-1])
    a = p.parse_args()
    apps, tools = a.build / "apps", a.build / "bench"
    counter = tools / "libomegah2csg_copy_counter.so"
    counter = counter if counter.is_file() else None
    cache = Path(__file__).with_name("_meshes")
    cache.mkdir(exist_ok=True)
    results = {}
    for n in (s for s in SIZES if s <= a.max_size):
        mesh = cache / f"box_{n}.osh"
        if not mesh.exists():
            subprocess.run(
                [tools / "omegah2csg_make_mesh", str(n), str(mesh)], check=True
            )
        results[f"openmc_box_{n}"] = bench(
            n, [apps / "convert2openmc", str(mesh)], counter
        )
    meshes = tomllib.loads(Path(__file__).with_name("meshes.toml").read_text())
    for name, entry in meshes.get("mesh", {}).items():
        results[f"degas2_{name}"] = bench(
            name, [apps / "convert2degas2", entry["path"]], counter
        )
    if "openmc_box_100000" in results and "openmc_box_10000" in results:
        ratio = (
            results["openmc_box_100000"]["total"] / results["openmc_box_10000"]["total"]
        )
        results["scaling_ratio_1e5_over_1e4"] = ratio
        print(f"scaling t(1e5)/t(1e4) = {ratio:.1f} (expected < 15)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    if a.baseline:
        base = json.loads(a.baseline.read_text())
        slow = [
            f"{case}/{stage}: {t:.3f}s vs {base[case][stage]:.3f}s"
            for case, stages in results.items()
            if isinstance(stages, dict) and case in base
            for stage, t in stages.items()
            if stage in base[case]
            and stage not in ("peak_rss_mb", "cross_space_copies", "bytes")
            and t > TOLERANCE * base[case][stage]
        ]
        if slow:
            sys.exit("slower than baseline:\n  " + "\n  ".join(slow))


if __name__ == "__main__":
    main()
