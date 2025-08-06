from typing import Any, List, Optional, Tuple, TypedDict, Union

import click
from bioio_ome_zarr.writers import DimTuple
from click import Context, Parameter

from ..converters.ome_zarr_converter import OmeZarrConverter


# ──────────────────────────────────────────────────────────────────────────────
# TypedDict for converter init kwargs
# ──────────────────────────────────────────────────────────────────────────────
class OmeZarrInitOptions(TypedDict, total=False):
    destination: str
    name: str
    scenes: Union[int, List[int]]
    tbatch: int
    level_scales: List[DimTuple]
    xy_scale: Tuple[float, ...]
    z_scale: Tuple[float, ...]
    chunk_memory_target: int
    dtype: str
    channel_names: List[str]
    auto_dask_cluster: bool


# ──────────────────────────────────────────────────────────────────────────────
# ParamTypes
# ──────────────────────────────────────────────────────────────────────────────
class DimTupleListType(click.ParamType):
    name = "level_scales"

    def convert(self, value: Any, param: Parameter, ctx: Context) -> List[DimTuple]:
        text = str(value)
        try:
            dims: List[DimTuple] = [
                tuple(float(x) for x in part.split(",")) for part in text.split(";")
            ]
        except Exception:
            self.fail(
                f"{value!r} is not a valid --level-scales value. "
                "Expected 't,c,z,y,x;...'",
                param,
                ctx,
            )
        return dims


class FloatListType(click.ParamType):
    name = "float_list"

    def convert(self, value: Any, param: Parameter, ctx: Context) -> Tuple[float, ...]:
        text = str(value)
        try:
            floats: Tuple[float, ...] = tuple(float(x) for x in text.split(","))
        except Exception:
            self.fail(
                f"{value!r} is not a valid float list.\n"
                "Expected comma-separated floats.",
                param,
                ctx,
            )
        return floats


class ScenesType(click.ParamType):
    name = "scenes"

    def convert(
        self, value: Any, param: Parameter, ctx: Context
    ) -> Union[int, List[int]]:
        try:
            parts = [int(x) for x in str(value).split(",")]
        except Exception:
            self.fail(
                f"{value!r} is not a valid --scenes value. "
                "Use a single index or comma-separated list.",
                param,
                ctx,
            )
        return parts[0] if len(parts) == 1 else parts


class ChannelNamesType(click.ParamType):
    name = "channel_names"

    def convert(self, value: Any, param: Parameter, ctx: Context) -> List[str]:
        labels = [c.strip() for c in str(value).split(",")]
        return labels


# ──────────────────────────────────────────────────────────────────────────────
# CLI definition
# ──────────────────────────────────────────────────────────────────────────────
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
    type=ScenesType(),
    default=None,
    help=("Which scene(s) to export: a single index or list (default: all scenes)"),
)
@click.option(
    "--tbatch",
    type=int,
    default=None,
    help="Number of timepoints per write batch",
)
@click.option(
    "--level-scales",
    type=DimTupleListType(),
    default=None,
    help=(
        "Semicolon-separated TCZYX tuples per level, " "e.g. '1,1,1,1,1;1,1,1,0.5,0.5'"
    ),
)
@click.option(
    "--xy-scale",
    type=FloatListType(),
    default=None,
    help="Comma-separated XY downsampling factors, e.g. '0.5,0.25'",
)
@click.option(
    "--z-scale",
    type=FloatListType(),
    default=None,
    help="Comma-separated Z downsampling factors, e.g. '1.0,0.5'",
)
@click.option(
    "--memory-target",
    type=int,
    default=None,
    help="Approximate bytes per Zarr chunk",
)
@click.option(
    "--dtype",
    default=None,
    help="Override output dtype, e.g. 'uint16'",
)
@click.option(
    "--channel-names",
    type=ChannelNamesType(),
    default=None,
    help="Comma-separated list of channel labels for metadata",
)
@click.option(
    "--auto-dask-cluster",
    is_flag=True,
    default=False,
    help="Create Dask cluster with 8 workers for parallel conversion",
)
def main(
    source: str,
    destination: str,
    name: Optional[str],
    scenes: Optional[Union[int, List[int]]],
    tbatch: Optional[int],
    level_scales: Optional[List[DimTuple]],
    xy_scale: Optional[Tuple[float, ...]],
    z_scale: Optional[Tuple[float, ...]],
    chunk_memory_target: Optional[int],
    dtype: Optional[str],
    channel_names: Optional[List[str]],
    auto_dask_cluster: bool,
) -> None:
    """
    Convert SOURCE to OME-Zarr stores under DESTINATION.
    """
    init_opts: OmeZarrInitOptions = {"destination": destination}

    if name is not None:
        init_opts["name"] = name
    if scenes is not None:
        init_opts["scenes"] = scenes
    if tbatch is not None:
        init_opts["tbatch"] = tbatch
    if level_scales is not None:
        init_opts["level_scales"] = level_scales
    if xy_scale is not None:
        init_opts["xy_scale"] = xy_scale
    if z_scale is not None:
        init_opts["z_scale"] = z_scale
    if chunk_memory_target is not None:
        init_opts["chunk_memory_target"] = chunk_memory_target
    if dtype is not None:
        init_opts["dtype"] = dtype
    if channel_names is not None:
        init_opts["channel_names"] = channel_names
    if auto_dask_cluster:
        init_opts["auto_dask_cluster"] = True

    conv = OmeZarrConverter(source=source, **init_opts)
    conv.convert()


if __name__ == "__main__":
    main()
