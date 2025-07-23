import json
from typing import Any, Dict, Optional, Tuple

import click

from .converters.batch_converter import BatchConverter


@click.command()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["csv", "dir", "list"], case_sensitive=False),
    required=True,
    help="Batch mode: 'csv' to read a CSV, 'dir' to walk a directory, or 'list' \n"
    "for explicit paths.",
)
@click.option(
    "--csv-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a CSV file describing conversion jobs (required in csv mode).",
)
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False),
    help="Root directory to scan for files (required in dir mode).",
)
@click.option(
    "--depth",
    "-d",
    type=int,
    default=1,
    show_default=True,
    help="Max recursion depth when scanning in dir mode.",
)
@click.option(
    "--pattern",
    "-p",
    default="*",
    show_default=True,
    help="Glob pattern to match files in dir mode.",
)
@click.option(
    "--paths",
    "-P",
    multiple=True,
    help="One or more explicit file paths (required in list mode).",
)
@click.option(
    "--converter",
    "-c",
    "converter_key",
    type=click.Choice(list(BatchConverter._CONVERTERS.keys())),
    default="ome-zarr",
    show_default=True,
    help="Which converter backend to use.",
)
@click.option(
    "--opt",
    "extra_opts",
    multiple=True,
    help="Extra converter init options as KEY=VALUE (VALUE may be JSON).",
)
def main(
    mode: str,
    csv_file: Optional[str],
    directory: Optional[str],
    depth: int,
    pattern: str,
    paths: Tuple[str, ...],
    converter_key: str,
    extra_opts: Tuple[str, ...],
) -> None:
    """
    Batch‐convert images via CSV, directory walk, or explicit path list.
    """
    # Parse extra_opts into a dict
    default_opts: Dict[str, Any] = {}
    for kv in extra_opts:
        if "=" not in kv:
            raise click.BadParameter(f"--opt must be KEY=VALUE, got '{kv}'")
        key, val = kv.split("=", 1)
        try:
            default_opts[key] = json.loads(val)
        except json.JSONDecodeError:
            default_opts[key] = val

    bc = BatchConverter(converter_key=converter_key, default_opts=default_opts)

    if mode.lower() == "csv":
        if not csv_file:
            raise click.BadParameter("--csv-file is required in csv mode")
        jobs = bc.from_csv(csv_file)
    elif mode.lower() == "dir":
        if not directory:
            raise click.BadParameter("--directory is required in dir mode")
        jobs = bc.from_directory(directory, max_depth=depth, pattern=pattern)
    else:  # list
        if not paths:
            raise click.BadParameter("--paths is required in list mode")
        jobs = bc.from_list(list(paths))

    click.echo(f"Discovered {len(jobs)} job(s), commencing conversion…")
    bc.run_jobs(jobs)
    click.echo("Batch conversion complete.")


if __name__ == "__main__":
    main()
