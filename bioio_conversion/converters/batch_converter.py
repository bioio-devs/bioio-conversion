import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd

from .ome_zarr_converter import OmeZarrConverter


class BatchConverter:
    """
    BatchConverter orchestrates bulk conversions of image files using
    a specified converter backend.

    Supports three input modes:
      - CSV-driven: each row defines a conversion job
      - Directory-driven: scan up to `max_depth` for matching files
      - List-driven: explicit list of file paths

    Default parameters for all jobs may be provided via `default_opts`.
    """

    # Map converter keys to classes
    _CONVERTERS: Dict[str, Type] = {
        "ome-zarr": OmeZarrConverter,
    }

    def __init__(
        self,
        *,
        converter_key: str = "ome-zarr",
        default_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the BatchConverter.

        Parameters
        ----------
        converter_key : str
            Key to select the converter backend (must exist in `_CONVERTERS`).
        default_opts : dict, optional
            Shared default options for each job (e.g. destination, tbatch, overwrite).
        """
        if converter_key not in self._CONVERTERS:
            raise KeyError(f"Unknown converter: {converter_key}")
        self.converter_cls = self._CONVERTERS[converter_key]
        self.default_opts = default_opts.copy() if default_opts else {}

    def from_csv(self, csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Parse a CSV file into a list of job option dicts.

        Each column maps to a converter parameter. Empty cells are skipped.
        Values that decode as JSON become native Python objects.
        """
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        jobs: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            opts = self.default_opts.copy()
            for col, val in row.items():
                if not val:
                    continue
                try:
                    parsed = json.loads(val)
                except json.JSONDecodeError:
                    parsed = val
                opts[col] = parsed
            jobs.append(opts)
        return jobs

    def from_directory(
        self,
        directory: Union[str, Path],
        *,
        max_depth: int = 0,
        pattern: str = "*",
    ) -> List[Dict[str, Any]]:
        """
        Recursively find files matching `pattern` up to `max_depth` levels.

        max_depth=0 → only top-level files
        max_depth=1 → include one subdirectory level, etc.
        """
        base = Path(directory)
        if not base.is_dir():
            raise ValueError(f"Not a directory: {base}")

        jobs: List[Dict[str, Any]] = []
        for path in base.rglob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(base)
            # depth = number of subfolders = len(parts) - 1
            if len(rel.parts) - 1 <= max_depth:
                opts = self.default_opts.copy()
                opts["source"] = str(path)
                jobs.append(opts)
        return jobs

    def from_list(
        self,
        paths: List[Union[str, Path]],
    ) -> List[Dict[str, Any]]:
        """
        Build jobs from an explicit list of file paths.

        Each path yields one job dict; default_opts are merged in.
        """
        jobs: List[Dict[str, Any]] = []
        for p in paths:
            opts = self.default_opts.copy()
            opts["source"] = str(p)
            jobs.append(opts)
        return jobs

    def run_jobs(
        self,
        jobs: List[Dict[str, Any]],
    ) -> None:
        """
        Execute each job: must include 'source'; merges defaults and job params.
        """
        for job in jobs:
            source = job.get("source")
            if not source:
                raise ValueError("Job missing 'source'")
            # Merge defaults and job params, excluding 'source'
            params = {k: v for k, v in job.items() if k != "source"}
            conv_opts = {**self.default_opts, **params}
            conv = self.converter_cls(source=source, **conv_opts)
            conv.convert()
