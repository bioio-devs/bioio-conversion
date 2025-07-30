from typing import Any, List, Optional, Tuple, Union

import click
from bioio_ome_zarr.writers import DimTuple

from .converters.ome_zarr_converter import OmeZarrConverter

_CONVERTERS = {
    "ome-zarr": OmeZarrConverter,
}


def parse_dimtuple_list(text: str) -> List[DimTuple]:
    """
    Parse semicolon-separated 5-tuples: "1,1,1,1,1;1,1,1,0.5,0.5"
    """
    out: List[DimTuple] = []
    for part in text.split(";"):
        nums = [float(x) for x in part.split(",")]
        if len(nums) != 5:
            raise click.BadParameter(
                f"Each level-scale must have exactly 5 values: '{part}'"
            )
        out.append(tuple(nums))  # type: ignore
    return out


def parse_float_list(text: str, name: str) -> Tuple[float, ...]:
    """
    Parse comma-separated floats: "0.5,0.25"
    """
    try:
        return tuple(float(x) for x in text.split(","))
    except ValueError:
        raise click.BadParameter(f"{name} must be comma-separated floats, got '{text}'")


def parse_scenes(text: str, total_scenes: int) -> Union[int, List[int]]:
    """
    Parse either:
      - "all" → -1 (the converter treats -1 as “all scenes”)
      - comma-separated ints → [i,j,…]
      - single int → i
    """
    if text.lower() == "all":
        return -1
    parts = text.split(",")
    if len(parts) == 1:
        try:
            return int(parts[0])
        except ValueError:
            raise click.BadParameter(
                f"Scenes must be 'all', a single integer, or comma-list; got '{text}'"
            )
    # multiple
    try:
        return [int(x) for x in parts]
    except ValueError:
        raise click.BadParameter(
            f"Scenes must be comma-separated integers or 'all'; got '{text}'"
        )


@click.command()
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--destination",
    "-d",
    required=True,
    type=click.Path(),
    help="Output directory for .ome.zarr stores",
)
@click.option(
    "--name",
    "-n",
    default=None,
    help="Base name for output stores (defaults to source stem)",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    default="ome-zarr",
    type=click.Choice(list(_CONVERTERS.keys())),
    help="Which converter to use",
)
@click.option(
    "--scenes",
    "-s",
    default="0",
    help="Scene(s) to export: 'all', a single index, or comma-list (e.g. 0,2)",
)
@click.option(
    "--tbatch", default=1, type=int, help="Number of timepoints per write batch"
)
@click.option(
    "--level-scales",
    default=None,
    help=(
        "Semicolon-separated list of TCZYX tuples for each level, "
        "e.g. '1,1,1,1,1;1,1,1,0.5,0.5'"
    ),
)
@click.option(
    "--xy-scale",
    default=None,
    help=(
        "Comma-separated XY downsampling factors (relative to original),\n"
        "e.g. '0.5,0.25'"
    ),
)
@click.option(
    "--z-scale",
    default=None,
    help=(
        "Comma-separated Z downsampling factors (relative to original),\n"
        "e.g. '1.0,0.5'"
    ),
)
@click.option(
    "--memory-target",
    default=16 * 1024 * 1024,
    type=int,
    help="Approximate bytes per Zarr chunk (default 16 MB)",
)
@click.option("--dtype", default=None, help="Override output dtype, e.g. 'uint16'")
@click.option(
    "--channel-names",
    default=None,
    help="Comma-separated list of channel labels to use in metadata",
)
def main(
    source: str,
    destination: str,
    name: Optional[str],
    fmt: str,
    scenes: str,
    tbatch: int,
    level_scales: Optional[str],
    xy_scale: Optional[str],
    z_scale: Optional[str],
    memory_target: int,
    dtype: Optional[str],
    channel_names: Optional[str],
) -> None:
    """
    Convert SOURCE → format under DESTINATION.

    You can export one scene, multiple scenes, or all scenes in a single run.
    """
    from bioio import BioImage

    total = len(BioImage(source).scenes)
    scenes_arg = parse_scenes(scenes, total)

    # 2) Build init kwargs
    init_opts: dict[str, Any] = {
        "destination": destination,
        "scenes": scenes_arg,
        "tbatch": tbatch,
        "memory_target": memory_target,
    }
    if name:
        init_opts["name"] = name
    if level_scales:
        init_opts["level_scales"] = parse_dimtuple_list(level_scales)
    if xy_scale:
        init_opts["xy_scale"] = parse_float_list(xy_scale, "xy-scale")
    if z_scale:
        init_opts["z_scale"] = parse_float_list(z_scale, "z-scale")
    if dtype:
        init_opts["dtype"] = dtype
    if channel_names:
        init_opts["channel_names"] = [c.strip() for c in channel_names.split(",")]

    # 3) Instantiate & run
    Converter = _CONVERTERS[fmt]
    conv = Converter(source=source, **init_opts)
    conv.convert()


if __name__ == "__main__":
    main()
