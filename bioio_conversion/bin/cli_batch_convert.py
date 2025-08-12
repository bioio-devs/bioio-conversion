import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click

from ..converters.batch_converter import BatchConverter


def parse_extra_opts(extra_opts: Tuple[str, ...]) -> Dict[str, Any]:
    """
    Parse KEY=VALUE pairs into a dict, JSON-decode values when possible.
    """
    opts: Dict[str, Any] = {}
    for kv in extra_opts:
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        try:
            opts[key] = json.loads(val)
        except json.JSONDecodeError:
            opts[key] = val
    return opts


@click.command()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["csv", "dir", "list"], case_sensitive=False),
    required=True,
    help="Batch mode: csv, dir, or list",
)
@click.option(
    "--csv-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to CSV file describing jobs (required in csv mode)",
)
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Root directory to scan (required in dir mode)",
)
@click.option(
    "--depth",
    "-d",
    type=int,
    default=0,
    show_default=True,
    help="Max recursion depth when scanning (dir mode)",
)
@click.option(
    "--pattern",
    "-p",
    default="*",
    show_default=True,
    help="Glob pattern to match files (dir mode)",
)
@click.option(
    "--paths",
    "-P",
    multiple=True,
    help="Explicit file paths (required in list mode)",
)
@click.option(
    "--opt",
    "-o",
    "extra_opts",
    multiple=True,
    help="Extra converter init options as KEY=VALUE",
)
def main(
    mode: str,
    csv_file: Optional[str],
    directory: Optional[str],
    depth: int,
    pattern: str,
    paths: Tuple[str, ...],
    extra_opts: Tuple[str, ...],
) -> None:
    """
    Batch-convert images via CSV, directory walk, or explicit list.
    """
    # Parse extra options
    default_opts: Dict[str, Any] = parse_extra_opts(extra_opts)

    bc = BatchConverter(default_opts=default_opts)

    mode_lower = mode.lower()
    if mode_lower == "csv":
        if csv_file is None:
            raise click.BadParameter("--csv-file is required in csv mode")
        jobs = bc.from_csv(Path(csv_file))
    elif mode_lower == "dir":
        if directory is None:
            raise click.BadParameter("--directory is required in dir mode")
        jobs = bc.from_directory(Path(directory), max_depth=depth, pattern=pattern)
    else:  # list
        if not paths:
            raise click.BadParameter("--paths is required in list mode")
        jobs = bc.from_list(list(paths))

    click.echo(f"Discovered {len(jobs)} job(s), commencing conversion…")
    bc.run_jobs(jobs)
    click.echo("Batch conversion complete.")


if __name__ == "__main__":
    main()
