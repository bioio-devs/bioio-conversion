
"""
Run ONE OME-Zarr conversion described in a JSON "data_source" file.

- Select the run via --run-index.
- The run MUST have a single key "source" (local or cloud path).
- Forwards OmeZarrConverter kwargs
- Outputs:
    - CSV: <out_root>/benchmark_results.csv
    - Per-run outputs: <out_root>/<dataset-name>/<out_subdir?>/<timestamp>/
"""

import argparse
import csv
import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from dask import config as dask_config

from bioio_conversion.converters import OmeZarrConverter

# CSV columns
FIELDNAMES = [
    "run_id",
    "dataset",
    "size_label",
    "scene",
    "source",
    "out_dir",
    "status",
    "error",
    "seconds",
    "started_at",
    "ended_at",
    "host",
    "cpu_logical",
    "cpu_physical",
    "mem_total_gb",
]

# Keys handled by this runner
RUNNER_KEYS = {
    "name",
    "size_label",
    "out_subdir",
    "source",
}

# ---- JSON → writer kwargs -----------------------------------------------------

def _coerce_tuple_like(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return tuple(_coerce_tuple_like(i) for i in x)
    return x


def _writer_kwargs_from_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build kwargs for OmeZarrConverter from a single 'run' dict.
    """
    kw: Dict[str, Any] = {k: v for k, v in run.items() if k not in RUNNER_KEYS}

    # New plural shape fields (per-level)
    for key in ("level_shapes", "chunk_shape", "shard_shape"):
        if key in kw and kw[key] is not None:
            kw[key] = _coerce_tuple_like(kw[key])


    for key in (
        "axes_names",
        "axes_types",
        "axes_units",
        "physical_pixel_size",
    ):
        if key in kw and kw[key] is not None:
            kw[key] = _coerce_tuple_like(kw[key])

    # scenes can be int or list[int]
    if "scenes" in kw:
        if isinstance(kw["scenes"], list):
            kw["scenes"] = [int(s) for s in kw["scenes"]]
        elif kw["scenes"] is not None:
            kw["scenes"] = int(kw["scenes"])

    # dtype as string (writer accepts str or np.dtype)
    if "dtype" in kw and kw["dtype"] is not None:
        kw["dtype"] = str(kw["dtype"])

    # numeric-ish args
    for key in ("memory_target", "tbatch", "start_t_src", "start_t_dest", "zarr_format"):
        if key in kw and kw[key] is not None:
            kw[key] = int(kw[key])

    return kw


# ---- Helpers ------------------------------------------------------------------

def _dest_dir_for_run(
    out_root: Path,
    dataset_name: str,
    out_subdir: Optional[str],
    run_id: str,
) -> Path:
    parts = [out_root, Path(dataset_name)]
    if out_subdir:
        parts.append(Path(str(out_subdir)))
    parts.append(Path(run_id))
    dest = Path(*parts)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _write_csv_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


# ---- Core (single run) --------------------------------------------------------

def _load_runs(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text() or "{}")
    runs = data.get("runs") or []
    if not isinstance(runs, list):
        raise ValueError("data_source JSON must contain a top-level 'runs' list.")
    return runs


def run_one(
    *,
    run: Dict[str, Any],
    out_root: Path,
    results_csv: Path,
) -> Dict[str, object]:
    dataset_name = str(run.get("name", "unnamed"))
    size_label = str(run.get("size_label", ""))
    out_subdir = run.get("out_subdir")

    # REQUIRED: single 'source'
    if not run.get("source"):
        raise ValueError("Run must include a 'source' path (local or cloud).")
    source = str(run["source"])

    writer_kwargs = _writer_kwargs_from_run(run)

    # Stamp + destination
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = _dest_dir_for_run(out_root, dataset_name, out_subdir, run_id)

    # Ensure destination + source
    writer_kwargs.setdefault("destination", str(dest_dir))
    writer_kwargs["source"] = source

    # Host/system info
    host = socket.gethostname()
    cpu_logical = psutil.cpu_count(logical=True) or 0
    cpu_physical = psutil.cpu_count(logical=False) or 0
    mem_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)

    started_at = datetime.now()
    status = "fail"
    err_msg = ""

    t0 = time.perf_counter()
    try:
        converter = OmeZarrConverter(**writer_kwargs)

        # Lazy reads run on Dask's threaded scheduler.
        with dask_config.set(scheduler="threads"):
            converter.convert()

        status = "ok"

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"

    finally:
        elapsed = round(time.perf_counter() - t0, 2)
        ended_at = datetime.now()

        scene_display = ""
        if "scenes" in writer_kwargs:
            scene_display = writer_kwargs["scenes"]

        row = {
            "run_id": run_id,
            "dataset": dataset_name,
            "size_label": size_label,
            "scene": scene_display,
            "source": source,
            "out_dir": str(dest_dir),
            "status": status,
            "error": err_msg,
            "seconds": elapsed,
            "started_at": started_at.isoformat(timespec="seconds"),
            "ended_at": ended_at.isoformat(timespec="seconds"),
            "host": host,
            "cpu_logical": cpu_logical,
            "cpu_physical": cpu_physical,
            "mem_total_gb": mem_total_gb,
        }

        _write_csv_row(results_csv, row)
        return row


# ---- CLI ---------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Run a single OME-Zarr conversion from a JSON data_source.")
    p.add_argument("--data-source", required=True, type=Path,
                   help="Path to JSON file with {'runs':[...]} specs (must contain 'source').")
    p.add_argument("--run-index", required=True, type=int,
                   help="Index within data_source.runs to execute (0-based).")
    p.add_argument("--out-root", type=Path, default=Path("./bench_out"),
                   help="Root directory for outputs (default: ./bench_out).")
    p.add_argument("--results-csv", type=Path, default=None,
                   help="Summary CSV path (default: <out-root>/benchmark_results.csv).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = args.results_csv or (out_root / "benchmark_results.csv")

    runs = _load_runs(args.data_source)
    if not (0 <= args.run_index < len(runs)):
        raise IndexError(f"--run-index {args.run_index} out of range (0..{len(runs)-1}).")

    row = run_one(
        run=runs[args.run_index],
        out_root=out_root,
        results_csv=results_csv,
    )

    print("\n=== Benchmark Summary ===")
    print(f"Wrote 1 row → {results_csv.resolve()}")
    print(f"Status: {row.get('status')}   Seconds: {row.get('seconds')}")
    print("Outputs root:", out_root.resolve())


if __name__ == "__main__":
    main()
