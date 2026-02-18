from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click

from ..converters.batch_converter import BatchConverter
from .ome_zarr_options import build_ome_zarr_init_opts, ome_zarr_options


@click.command()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["csv", "dir", "list"], case_sensitive=False),
    required=True,
    help=(
        "Batch mode: read jobs from a CSV, scan a directory, or use an "
        "explicit list of paths."
    ),
)
@click.option(
    "--csv-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to CSV describing jobs (required when --mode csv).",
)
@click.option(
    "--directory",
    "--dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Root directory to scan (required when --mode dir).",
)
@click.option(
    "--depth",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Max recursion depth when scanning directories (dir mode). "
        "0 = only top-level files."
    ),
)
@click.option(
    "--pattern",
    default="*",
    show_default=True,
    help="Glob pattern used when scanning directories (dir mode).",
)
@click.option(
    "--paths",
    multiple=True,
    help=("Explicit input file paths (required when --mode list). " "Repeatable."),
)
@ome_zarr_options(require_destination=True)
def main(
    mode: str,
    csv_file: Optional[str],
    directory: Optional[str],
    depth: int,
    pattern: str,
    paths: Tuple[str, ...],
    **kwargs: Any,
) -> None:
    """
    Batch-convert images to OME-Zarr.

    This CLI supports three job-discovery modes:

    - csv: Read one job per row from a CSV file. Each column name maps to an
      OmeZarrConverter init argument (for example source, destination,
      scenes, tbatch). Values are parsed by the CSV loader.

    - dir: Recursively scan a directory for files matching --pattern, up to
      --depth levels deep. Each discovered file becomes a job.

    - list: Convert an explicit list of --paths provided on the command
      line.

    Converter options (destination, chunking, pyramid settings, channels,
    axes, and so on) are provided via the shared ome_zarr_options flags.
    These flags are converted into OmeZarrConverter init kwargs via
    build_ome_zarr_init_opts and applied as defaults for all jobs.

    Examples:

    \b
      # CSV mode
      bioio-batch-convert --mode csv --csv-file jobs.csv \\
          --destination out --tbatch 1

    \b
      # Directory mode
      bioio-batch-convert --mode dir --directory data --pattern "*.czi" \\
          --depth 1 --destination out

    \b
      # List mode
      bioio-batch-convert --mode list --paths a.ome.tiff b.ome.tiff \\
          --destination out
    """
    try:
        init_opts_typed = build_ome_zarr_init_opts(**kwargs)
        default_opts: Dict[str, Any] = dict(init_opts_typed)

        bc = BatchConverter(default_opts=default_opts)

        mode_lower = mode.lower()
        if mode_lower == "csv":
            if csv_file is None:
                raise click.UsageError("--csv-file is required when --mode csv")
            jobs = bc.from_csv(Path(csv_file))

        elif mode_lower == "dir":
            if directory is None:
                raise click.UsageError("--directory/--dir is required when --mode dir")
            jobs = bc.from_directory(Path(directory), max_depth=depth, pattern=pattern)

        else:  # list
            if not paths:
                raise click.UsageError(
                    "At least one --paths is required when --mode list"
                )
            jobs = bc.from_list(list(paths))

        click.echo(f"Discovered {len(jobs)} job(s), commencing conversion…")
        bc.run_jobs(jobs)
        click.echo("Batch conversion complete.")

    except KeyboardInterrupt:
        raise click.Abort()

    except click.ClickException:
        raise

    except Exception as e:
        raise click.ClickException(f"Conversion failed: {e}")
