from typing import Any, List, Optional, Tuple, Union

import click
from bioio_ome_zarr.writers import DimTuple

from .converters.ome_zarr_converter import OmeZarrConverter


def parse_dimtuple_list(text: str) -> List[DimTuple]:
    """
    Parse semicolon-separated 5-tuples: "1,1,1,1,1;1,1,1,0.5,0.5"
    """
    return [tuple(float(x) for x in part.split(",")) for part in text.split(";")]


def parse_float_list(text: str) -> Tuple[float, ...]:
    """
    Parse comma-separated floats: "0.5,0.25"
    """
    return tuple(float(x) for x in text.split(","))


def parse_scenes(text: str) -> Union[int, List[int]]:
    """
    Parse scene specifier: 'all' → -1, '0' → 0, '0,2,3' → [0, 2, 3]
    """
    if text.lower() == "all":
        return -1
    parts = text.split(",")
    if len(parts) > 1:
        return [int(x) for x in parts]
    return int(parts[0])


def parse_channel_names(text: str) -> List[str]:
    """
    Parse comma-separated channel labels into a list of strings.
    """
    return [c.strip() for c in text.split(",")]


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
    "--scenes",
    "-s",
    default=None,
    help="Scene(s) to export: 'all', a single index, or comma-list",
)
@click.option(
    "--tbatch",
    default=None,
    type=int,
    help="Number of timepoints per write batch",
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
    help="Comma-separated XY downsampling factors, e.g. '0.5,0.25'",
)
@click.option(
    "--z-scale",
    default=None,
    help="Comma-separated Z downsampling factors, e.g. '1.0,0.5'",
)
@click.option(
    "--memory-target",
    default=None,
    type=int,
    help="Approximate bytes per Zarr chunk",
)
@click.option(
    "--dtype",
    default=None,
    help="Override output dtype, e.g. 'uint16'",
)
@click.option(
    "--channel-names",
    default=None,
    help="Comma-separated list of channel labels to use in metadata",
)
@click.option(
    "--auto-dask-cluster",
    is_flag=True,
    default=False,
    help="Create Dask cluster with 8 workers for Dask computation.",
)
def main(
    source: str,
    destination: str,
    name: Optional[str],
    scenes: Optional[str],
    tbatch: Optional[int],
    level_scales: Optional[str],
    xy_scale: Optional[str],
    z_scale: Optional[str],
    memory_target: Optional[int],
    dtype: Optional[str],
    channel_names: Optional[str],
    auto_dask_cluster: bool,
) -> None:
    """
    Convert SOURCE to OME-Zarr stores under DESTINATION.
    """
    init_opts: dict[str, Any] = {}

    init_opts["destination"] = destination
    if name is not None:
        init_opts["name"] = name
    if scenes is not None:
        init_opts["scenes"] = parse_scenes(scenes)
    if tbatch is not None:
        init_opts["tbatch"] = tbatch
    if level_scales is not None:
        init_opts["level_scales"] = parse_dimtuple_list(level_scales)
    if xy_scale is not None:
        init_opts["xy_scale"] = parse_float_list(xy_scale)
    if z_scale is not None:
        init_opts["z_scale"] = parse_float_list(z_scale)
    if memory_target is not None:
        init_opts["memory_target"] = memory_target
    if dtype is not None:
        init_opts["dtype"] = dtype
    if channel_names is not None:
        init_opts["channel_names"] = parse_channel_names(channel_names)
    if auto_dask_cluster:
        init_opts["auto_dask_cluster"] = True

    conv = OmeZarrConverter(source=source, **init_opts)
    conv.convert()


if __name__ == "__main__":
    main()
