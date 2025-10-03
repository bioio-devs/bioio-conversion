from typing import Any, List, Optional, Sequence, Tuple, TypedDict, Union

import click
from bioio_ome_zarr.writers import Channel
from bioio_ome_zarr.writers.ome_zarr_writer import MultiResolutionShapeSpec
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
    start_T_src: int
    start_T_dest: int
    level_shapes: MultiResolutionShapeSpec
    num_levels: int
    downsample_z: bool
    chunk_shape: MultiResolutionShapeSpec
    memory_target: int
    shard_shape: MultiResolutionShapeSpec
    dtype: str
    channels: List[Channel]
    physical_pixel_size: List[float]
    zarr_format: int


# ──────────────────────────────────────────────────────────────────────────────
# ParamTypes
# ──────────────────────────────────────────────────────────────────────────────


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


class IntListType(click.ParamType):
    name = "int_list"

    def convert(self, value: Any, param: Parameter, ctx: Context) -> Tuple[int, ...]:
        text = str(value)
        try:
            ints: Tuple[int, ...]
            if text.strip() == "":
                ints = tuple()
            else:
                ints = tuple(int(x) for x in text.split(","))
        except Exception:
            self.fail(
                f"{value} is not a valid integer list. Example: (1, 1, 16, 256, 256)",
                param,
                ctx,
            )
        return ints


class IntTupleListType(click.ParamType):
    name = "int_tuple_list"

    def convert(
        self, value: Any, param: Parameter, ctx: Context
    ) -> List[Tuple[int, ...]]:
        text = str(value)
        try:
            int_tuples: List[Tuple[int, ...]]
            if text.strip() == "":
                int_tuples = []
            else:
                int_tuples = [
                    tuple(int(x) for x in part.split(",")) for part in text.split(";")
                ]
        except Exception:
            self.fail(
                f"{value!r} is not a valid semicolon-separated list of int tuples. "
                "Example: '1,1,16,256,256;1,1,16,128,128'",
                param,
                ctx,
            )
        return int_tuples


class ScenesType(click.ParamType):
    name = "scenes"

    def convert(
        self, value: Any, param: Parameter, ctx: Context
    ) -> Union[int, List[int]]:
        text = str(value).strip()
        try:
            parts = [int(x) for x in text.split(",")]
        except Exception:
            self.fail(
                f"{value!r} is not a valid --scenes value. "
                "Use a single index or comma-separated list (e.g. '0,2').",
                param,
                ctx,
            )
        return parts[0] if len(parts) == 1 else parts


class StrListType(click.ParamType):
    name = "str_list"

    def convert(self, value: Any, param: Parameter, ctx: Context) -> List[str]:
        return [c.strip() for c in str(value).split(",") if c.strip()]


class BoolListType(click.ParamType):
    """Parse comma-separated booleans like 'true,false,1,0,yes,no'."""

    name = "bool_list"
    TRUE = {"1", "true", "t", "yes", "y", "on"}
    FALSE = {"0", "false", "f", "no", "n", "off"}

    def convert(self, value: Any, param: Parameter, ctx: Context) -> Tuple[bool, ...]:
        text = str(value)
        try:
            vals: Tuple[bool, ...]
            if text.strip() == "":
                vals = tuple()
            else:
                parsed: List[bool] = []
                for tok in text.split(","):
                    s = tok.strip().lower()
                    if s in self.TRUE:
                        parsed.append(True)
                    elif s in self.FALSE:
                        parsed.append(False)
                    else:
                        raise ValueError(s)
                vals = tuple(parsed)
        except Exception:
            self.fail(
                f"{value!r} is not a valid boolean list. Use true/false or 1/0.",
                param,
                ctx,
            )
        return vals


def _get(seq: Optional[Sequence[Any]], idx: int, default: Any) -> Any:
    return seq[idx] if seq is not None and idx < len(seq) else default


def _build_channels(
    labels: List[str],
    colors: Optional[List[str]],
    actives: Optional[Tuple[bool, ...]],
    coefs: Optional[Tuple[float, ...]],
    families: Optional[List[str]],
    inverted: Optional[Tuple[bool, ...]],
    w_min: Optional[Tuple[int, ...]],
    w_max: Optional[Tuple[int, ...]],
    w_start: Optional[Tuple[int, ...]],
    w_end: Optional[Tuple[int, ...]],
) -> List[Channel]:
    channels: List[Channel] = []
    # Determine if any non-required channel attributes were provided at all
    any_optional = any(
        v is not None
        for v in (actives, coefs, families, inverted, w_min, w_max, w_start, w_end)
    )

    for i, label in enumerate(labels):
        # Always supply minimal required args
        base_color = _get(colors, i, "#FFFFFF")
        ch_kwargs: dict = {"label": label, "color": base_color}

        # Only pass optional kwargs if the user provided that group of values
        if any_optional:
            if actives is not None and i < len(actives):
                ch_kwargs["active"] = bool(actives[i])
            if coefs is not None and i < len(coefs):
                ch_kwargs["coefficient"] = float(coefs[i])
            if families is not None and i < len(families):
                ch_kwargs["family"] = families[i]
            if inverted is not None and i < len(inverted):
                ch_kwargs["inverted"] = bool(inverted[i])

            # Window: only include if any window tokens were provided;
            # fill missing pieces with Channel defaults.
            if any(v is not None for v in (w_min, w_max, w_start, w_end)):
                ch_kwargs["window"] = {
                    "min": _get(w_min, i, 0),
                    "max": _get(w_max, i, 255),
                    "start": _get(w_start, i, 0),
                    "end": _get(w_end, i, 255),
                }

        channels.append(Channel(**ch_kwargs))
    return channels


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
    help="Which scene(s) to export, e.g. '0' or '0,2'. Default: all",
)
@click.option(
    "--tbatch", type=int, default=None, help="Number of timepoints per write batch"
)
@click.option(
    "--start-t-src",
    type=int,
    default=None,
    help="Source T index at which to begin reading (maps to writer.start_T_src)",
)
@click.option(
    "--start-t-dest",
    type=int,
    default=None,
    help="Destination T index at which to begin writing (maps to writer.start_T_dest)",
)
# --- multiscale control ---
@click.option(
    "--level-shapes",
    type=IntTupleListType(),
    default=None,
    help=(
        "Semicolon-separated per-level SHAPES (ints), level 0 first. "
        "Each tuple length must match native axes. "
        "Example: '2,3,5,512,512;2,3,5,256,256;2,3,5,128,128'. "
        "If provided, overrides --num-levels/--downsample-z."
    ),
)
@click.option(
    "--num-levels",
    type=int,
    default=None,
    help="Total multiscale levels (>=1). If provided (and --level-shapes not set), "
    "build a half-pyramid: X/Y always 0.5^level; add --downsample-z to also "
    "downsample Z when present.",
)
@click.option(
    "--downsample-z",
    is_flag=True,
    default=False,
    help="With --num-levels, also half Z per level when a Z axis exists.",
)
# --- chunking ---
@click.option(
    "--chunk-shape",
    type=IntListType(),
    default=None,
    help="Single chunk shape tuple, e.g. '1,1,16,256,256'.",
)
@click.option(
    "--chunk-shape-per-level",
    type=IntTupleListType(),
    default=None,
    help="Per-level chunk shapes, e.g. '1,1,16,256,256;1,1,16,128,128'.",
)
@click.option(
    "--shard-shape",
    type=IntListType(),
    default=None,
    help="(Zarr v3) Single shard shape, e.g. '1,1,128,1024,1024'.",
)
@click.option(
    "--shard-shape-per-level",
    type=IntTupleListType(),
    default=None,
    help="(Zarr v3) Per-level shard shapes, semicolon-separated int tuples.",
)
@click.option(
    "--memory-target",
    type=int,
    default=None,
    help="generate and use per-level chunk shapes from this in-memory byte target.",
)
# --- metadata & writer ---
@click.option("--dtype", default=None)
@click.option("--physical-pixel-sizes", type=FloatListType(), default=None)
@click.option(
    "--zarr-format",
    type=click.Choice(["2", "3"], case_sensitive=False),
    default=None,  # Optional: let the writer default when omitted
    help="Target Zarr format (2=NGFF 0.4, 3=NGFF 0.5). If None, use writer default ",
)
# --- Channel (full access) ---
@click.option(
    "--channel-labels",
    type=StrListType(),
    default=None,
    help="Comma-separated channel labels. If provided, Channel[] will be built.",
)
@click.option(
    "--channel-colors",
    type=StrListType(),
    default=None,
    help="Comma-separated channel colors (e.g. '#FF00FF,red,#00FF00').",
)
@click.option(
    "--channel-actives",
    type=BoolListType(),
    default=None,
    help="Comma-separated booleans for channel visibility, e.g. 'true,false'.",
)
@click.option(
    "--channel-coefficients",
    type=FloatListType(),
    default=None,
    help="Comma-separated floats for coefficients, e.g. '1,0.8,1'.",
)
@click.option(
    "--channel-families",
    type=StrListType(),
    default=None,
    help="Comma-separated intensity families (e.g. 'linear,sRGB').",
)
@click.option(
    "--channel-inverted",
    type=BoolListType(),
    default=None,
    help="Comma-separated booleans for inversion flags.",
)
@click.option(
    "--channel-window-min",
    type=IntListType(),
    default=None,
    help="Comma-separated ints for window.min per channel.",
)
@click.option(
    "--channel-window-max",
    type=IntListType(),
    default=None,
    help="Comma-separated ints for window.max per channel.",
)
@click.option(
    "--channel-window-start",
    type=IntListType(),
    default=None,
    help="Comma-separated ints for window.start per channel.",
)
@click.option(
    "--channel-window-end",
    type=IntListType(),
    default=None,
    help="Comma-separated ints for window.end per channel.",
)
def main(
    source: str,
    destination: str,
    name: Optional[str],
    scenes: Optional[Union[int, List[int]]],
    tbatch: Optional[int],
    start_t_src: Optional[int],
    start_t_dest: Optional[int],
    level_shapes: Optional[List[Tuple[int, ...]]],
    num_levels: Optional[int],
    downsample_z: bool,
    chunk_shape: Optional[Tuple[int, ...]],
    chunk_shape_per_level: Optional[List[Tuple[int, ...]]],
    shard_shape: Optional[Tuple[int, ...]],
    shard_shape_per_level: Optional[List[Tuple[int, ...]]],
    memory_target: Optional[int],
    dtype: Optional[str],
    physical_pixel_sizes: Optional[Tuple[float, ...]],
    zarr_format: Optional[str],
    channel_labels: Optional[List[str]],
    channel_colors: Optional[List[str]],
    channel_actives: Optional[Tuple[bool, ...]],
    channel_coefficients: Optional[Tuple[float, ...]],
    channel_families: Optional[List[str]],
    channel_inverted: Optional[Tuple[bool, ...]],
    channel_window_min: Optional[Tuple[int, ...]],
    channel_window_max: Optional[Tuple[int, ...]],
    channel_window_start: Optional[Tuple[int, ...]],
    channel_window_end: Optional[Tuple[int, ...]],
) -> None:
    init_opts: OmeZarrInitOptions = {"destination": destination}

    if zarr_format is not None:
        init_opts["zarr_format"] = int(zarr_format)
    if name is not None:
        init_opts["name"] = name
    if scenes is not None:
        init_opts["scenes"] = scenes
    if tbatch is not None:
        init_opts["tbatch"] = tbatch
    if start_t_src is not None:
        init_opts["start_T_src"] = start_t_src
    if start_t_dest is not None:
        init_opts["start_T_dest"] = start_t_dest
    if level_shapes and len(level_shapes) > 0:
        init_opts["level_shapes"] = level_shapes
    elif num_levels is not None:
        init_opts["num_levels"] = num_levels
        if downsample_z:
            init_opts["downsample_z"] = True
    if chunk_shape_per_level and len(chunk_shape_per_level) > 0:
        init_opts["chunk_shape"] = chunk_shape_per_level
    elif chunk_shape:
        init_opts["chunk_shape"] = chunk_shape

    if shard_shape_per_level and len(shard_shape_per_level) > 0:
        init_opts["shard_shape"] = shard_shape_per_level
    elif shard_shape:
        init_opts["shard_shape"] = shard_shape

    if memory_target is not None:
        init_opts["memory_target"] = memory_target
    if dtype is not None:
        init_opts["dtype"] = dtype
    if physical_pixel_sizes:
        init_opts["physical_pixel_size"] = list(physical_pixel_sizes)

    # Channels (full access; use defaults when options omitted)
    if channel_labels is not None and len(channel_labels) > 0:
        init_opts["channels"] = _build_channels(
            labels=channel_labels,
            colors=channel_colors,
            actives=channel_actives,
            coefs=channel_coefficients,
            families=channel_families,
            inverted=channel_inverted,
            w_min=channel_window_min,
            w_max=channel_window_max,
            w_start=channel_window_start,
            w_end=channel_window_end,
        )

    conv = OmeZarrConverter(source=source, **init_opts)
    conv.convert()


if __name__ == "__main__":
    main()
