# bioio_conversion/conversion_cli.py

from typing import Any, List, Optional

import click
from bioio_ome_zarr.writers import DimTuple

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter

_CONVERTERS = {
    "ome-zarr": OmeZarrConverter,
}


def parse_level_scales(text: str) -> List[DimTuple]:
    """
    Parse semicolon-separated 5-tuples: "1,1,1,1,1;1,1,1,0.5,0.5"
    """
    levels: List[DimTuple] = []
    for part in text.split(";"):
        vals = [float(x) for x in part.split(",")]
        if len(vals) != 5:
            raise click.BadParameter(f"Each level-scale must have 5 values: '{part}'")
        levels.append(tuple(vals))  # type: ignore
    return levels


@click.command()
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--destination", "-d", required=True, type=click.Path(), help="Output folder"
)
@click.option(
    "--name", "-n", default=None, help="Output name (defaults to source stem)"
)
@click.option(
    "--format",
    "-f",
    "fmt",
    default="ome-zarr",
    type=click.Choice(list(_CONVERTERS.keys())),
    help="Converter format",
)
@click.option("--scene", type=int, default=0, help="Scene index to convert (default 0)")
@click.option(
    "--overwrite/--no-overwrite", default=False, help="Overwrite existing output"
)
@click.option(
    "--tbatch", default=1, type=int, help="Time-batch size for streaming writes"
)
@click.option(
    "--level-scales",
    default=None,
    help=(
        "Semicolon-separated list of 5-tuples for per-level scales, "
        "e.g. '1,1,1,1,1;1,1,1,0.5,0.5'"
    ),
)
@click.option("--dtype", default=None, help="Override output data type (e.g. 'uint16')")
@click.option(
    "--channel-names",
    default=None,
    help="Comma-separated list of channel names to use in output metadata",
)
def main(
    source: str,
    destination: str,
    name: Optional[str],
    fmt: str,
    scene: int,
    overwrite: bool,
    tbatch: int,
    level_scales: Optional[str],
    dtype: Optional[str],
    channel_names: Optional[str],
) -> None:
    """
    Convert SOURCE to the specified format under DESTINATION.
    """
    # Build init options
    init_opts: dict[str, Any] = {
        "overwrite": overwrite,
        "tbatch": tbatch,
    }
    if name:
        init_opts["name"] = name
    if level_scales:
        init_opts["level_scales"] = parse_level_scales(level_scales)
    if dtype:
        init_opts["dtype"] = dtype
    if channel_names:
        init_opts["channel_names"] = channel_names.split(",")

    # Instantiate converter
    Converter = _CONVERTERS[fmt]
    conv = Converter(
        source=source,
        destination=destination,
        scene=scene,
        **init_opts,
    )
    conv.convert()


if __name__ == "__main__":
    main()
