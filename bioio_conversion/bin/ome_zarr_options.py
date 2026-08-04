import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import click
from bioio_ome_zarr.writers import Channel
from bioio_ome_zarr.writers.ome_zarr_writer import MultiResolutionShapeSpec
from click import Context, Parameter


class OmeZarrInitOptions(TypedDict, total=False):
    """
    The subset of `OmeZarrConverter.__init__` kwargs that can be provided
    by the CLI.
    """

    axes_names: List[str]
    axes_types: List[str]
    axes_units: List[Optional[str]]
    destination: str
    name: str
    scenes: Union[int, List[int]]
    tbatch: int
    start_t_src: int
    start_t_dest: int

    # pyramid / multiscale
    level_shapes: MultiResolutionShapeSpec
    num_levels: int
    downsample_z: bool

    # chunking / sharding
    chunk_shape: MultiResolutionShapeSpec
    memory_target: int
    shard_shape: MultiResolutionShapeSpec

    # data / metadata
    dtype: str
    channels: List[Channel]
    physical_pixel_size: List[float]
    zarr_format: int

    # provenance
    include_provenance: bool
    provenance_reader_kwargs: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# ParamTypes
# ──────────────────────────────────────────────────────────────────────────────


class FloatListType(click.ParamType):
    """Parse '1.0,2.0,3.5' -> (1.0, 2.0, 3.5)"""

    name = "float_list"

    def convert(
        self,
        value: Any,
        param: Parameter,
        ctx: Context,
    ) -> Tuple[float, ...]:
        text = str(value)
        try:
            return tuple(float(x) for x in text.split(","))
        except Exception:
            self.fail(
                f"{value!r} is not a valid float list (comma-separated).",
                param,
                ctx,
            )
            # For type checkers: self.fail always raises.
            assert False, "unreachable"


class IntListType(click.ParamType):
    """Parse '1,2,3' -> (1, 2, 3). Empty string -> ()."""

    name = "int_list"

    def convert(
        self,
        value: Any,
        param: Parameter,
        ctx: Context,
    ) -> Tuple[int, ...]:
        text = str(value).strip()
        try:
            return tuple() if text == "" else tuple(int(x) for x in text.split(","))
        except Exception:
            self.fail(
                f"{value!r} is not a valid int list (comma-separated).",
                param,
                ctx,
            )
            assert False, "unreachable"


class IntTupleListType(click.ParamType):
    """
    Parse per-level tuples like:
        '1,1,16,256,256;1,1,16,128,128'
    ->  [(1,1,16,256,256), (1,1,16,128,128)]

    Empty string -> [].
    """

    name = "int_tuple_list"

    def convert(
        self,
        value: Any,
        param: Parameter,
        ctx: Context,
    ) -> List[Tuple[int, ...]]:
        text = str(value).strip()
        if text == "":
            return []

        try:
            return [tuple(int(x) for x in part.split(",")) for part in text.split(";")]
        except Exception:
            message = (
                f"{value!r} is not a valid list of int tuples separated by "
                "semicolons. Example: "
                "'1,1,16,256,256;1,1,16,128,128'."
            )
            self.fail(message, param, ctx)
            raise AssertionError("unreachable")


class ScenesType(click.ParamType):
    """
    Parse scenes selection:
      '0'   -> 0
      '0,2' -> [0, 2]
    """

    name = "scenes"

    def convert(
        self,
        value: Any,
        param: Parameter,
        ctx: Context,
    ) -> Union[int, List[int]]:
        text = str(value).strip()
        try:
            parts = [int(x) for x in text.split(",")]
        except Exception:
            message = (
                f"{value!r} is not a valid --scenes value. Use a single index "
                "like 0 or a comma-separated list like 0,2."
            )
            self.fail(message, param, ctx)
            raise AssertionError("unreachable")

        return parts[0] if len(parts) == 1 else parts


class StrListType(click.ParamType):
    """
    Parse 'a,b,c' -> ['a','b','c'] (strips whitespace, drops empty tokens).
    """

    name = "str_list"

    def convert(
        self,
        value: Any,
        param: Parameter,
        ctx: Context,
    ) -> List[str]:
        return [c.strip() for c in str(value).split(",") if c.strip()]


class BoolListType(click.ParamType):
    """
    Parse comma-separated booleans:

      'true,false,1,0,yes,no,on,off'
      -> (True, False, True, False, True, False, True, False)
    """

    name = "bool_list"

    TRUE = {"1", "true", "t", "yes", "y", "on"}
    FALSE = {"0", "false", "f", "no", "n", "off"}

    def convert(
        self,
        value: Any,
        param: Parameter,
        ctx: Context,
    ) -> Tuple[bool, ...]:
        text = str(value).strip()
        if text == "":
            return tuple()

        out: List[bool] = []
        for tok in text.split(","):
            s = tok.strip().lower()
            if s in self.TRUE:
                out.append(True)
            elif s in self.FALSE:
                out.append(False)
            else:
                self.fail(
                    f"{value!r} is not a valid boolean list.",
                    param,
                    ctx,
                )
                assert False, "unreachable"
        return tuple(out)


class OptionalStrListType(click.ParamType):
    """
    Parse comma-separated strings where blanks/'none'/'null' become None.

    Example:
      's,,um,um,um' -> ['s', None, 'um', 'um', 'um']
    """

    name = "optional_str_list"
    NONE_TOKENS = {"", "none", "null", "nil"}

    def convert(
        self,
        value: Any,
        param: Parameter,
        ctx: Context,
    ) -> List[Optional[str]]:
        parts = [p.strip() for p in str(value).split(",")]
        out: List[Optional[str]] = []
        for p in parts:
            out.append(None if p.lower() in self.NONE_TOKENS else p)
        return out


class JsonDictType(click.ParamType):
    """
    Parse a JSON object into a dict:
      '{"plate": "96"}' -> {"plate": "96"}
    """

    name = "json_dict"

    def convert(
        self,
        value: Any,
        param: Parameter,
        ctx: Context,
    ) -> Dict[str, Any]:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            self.fail(f"{value!r} is not valid JSON ({exc}).", param, ctx)

        if not isinstance(parsed, dict):
            message = (
                f"{value!r} must be a JSON object of reader kwargs, for example "
                '\'{"plate": "96"}\'.'
            )
            self.fail(message, param, ctx)

        return parsed


def _get(
    seq: Optional[Sequence[Any]],
    idx: int,
    default: Any,
) -> Any:
    """Safe index helper for per-channel option lists."""
    return seq[idx] if seq is not None and idx < len(seq) else default


def build_channels(
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
    """
    Build `Channel[]` from per-channel CLI options.
    """
    channels: List[Channel] = []

    any_optional = any(
        v is not None
        for v in (
            actives,
            coefs,
            families,
            inverted,
            w_min,
            w_max,
            w_start,
            w_end,
        )
    )

    for i, label in enumerate(labels):
        ch_kwargs: Dict[str, Any] = {
            "label": label,
            "color": _get(colors, i, "#FFFFFF"),
        }

        if any_optional:
            # Only apply optional fields where values were supplied
            if actives is not None and i < len(actives):
                ch_kwargs["active"] = bool(actives[i])
            if coefs is not None and i < len(coefs):
                ch_kwargs["coefficient"] = float(coefs[i])
            if families is not None and i < len(families):
                ch_kwargs["family"] = families[i]
            if inverted is not None and i < len(inverted):
                ch_kwargs["inverted"] = bool(inverted[i])

            if any(v is not None for v in (w_min, w_max, w_start, w_end)):
                ch_kwargs["window"] = {
                    "min": _get(w_min, i, 0),
                    "max": _get(w_max, i, 255),
                    "start": _get(w_start, i, 0),
                    "end": _get(w_end, i, 255),
                }

        channels.append(Channel(**ch_kwargs))

    return channels


F = TypeVar("F", bound=Callable[..., Any])


def ome_zarr_options(
    *,
    require_destination: bool,
) -> Callable[[F], F]:
    """
    Attach the shared OME-Zarr converter flags to a Click command.

    These options are translated into `OmeZarrConverter.__init__` kwargs
    via `build_ome_zarr_init_opts(...)` and can be reused across CLIs
    (single-file or batch).
    """

    def _decorator(fn: F) -> F:
        opts = [
            # ── Axis controls ────────────────────────────────────────────
            click.option(
                "--axes-names",
                type=StrListType(),
                default=None,
                help=(
                    "Comma-separated axis names to write, in native axis "
                    "order. Example: 't,c,z,y,x'. If omitted, use names "
                    "derived from the reader."
                ),
            ),
            click.option(
                "--axes-types",
                type=StrListType(),
                default=None,
                help=(
                    "Comma-separated axis semantic types, one per axis name. "
                    "Typical values include 'time', 'channel', and 'space'. "
                    "Example: 'time,channel,space,space,space'."
                ),
            ),
            click.option(
                "--axes-units",
                type=OptionalStrListType(),
                default=None,
                help=(
                    "Comma-separated axis units, in the same order as "
                    "--axes-names. Use blank or 'none'/'null' for missing "
                    "units. Example for (t,c,z,y,x): 's,,um,um,um'."
                ),
            ),
            # ── Channel controls (only used when --channel-labels is set) ─
            click.option(
                "--channel-labels",
                type=StrListType(),
                default=None,
                help=(
                    "Comma-separated channel labels. If provided, a Channel[] "
                    "definition is built and written into the OME-Zarr "
                    "metadata. Example: 'DAPI,GFP,TRITC'."
                ),
            ),
            click.option(
                "--channel-colors",
                type=StrListType(),
                default=None,
                help=(
                    "Comma-separated channel colors matching --channel-labels. "
                    "Values may be CSS color names or hex codes. Example: "
                    "'#0000FF,#00FF00,#FF0000'."
                ),
            ),
            click.option(
                "--channel-actives",
                type=BoolListType(),
                default=None,
                help=(
                    "Comma-separated booleans controlling channel visibility. "
                    "Example: 'true,true,false' to hide the third channel."
                ),
            ),
            click.option(
                "--channel-coefficients",
                type=FloatListType(),
                default=None,
                help=(
                    "Comma-separated floats for per-channel intensity "
                    "coefficients. Example: '1,0.8,1.2'."
                ),
            ),
            click.option(
                "--channel-families",
                type=StrListType(),
                default=None,
                help=(
                    "Comma-separated intensity families per channel "
                    "(e.g. 'linear', 'sRGB'). Example: 'linear,sRGB,sRGB'."
                ),
            ),
            click.option(
                "--channel-inverted",
                type=BoolListType(),
                default=None,
                help=(
                    "Comma-separated booleans for inverted display per channel. "
                    "Example: 'false,true,false'."
                ),
            ),
            click.option(
                "--channel-window-min",
                type=IntListType(),
                default=None,
                help=(
                    "Comma-separated ints for window.min per channel. Only "
                    "used when any window value is provided."
                ),
            ),
            click.option(
                "--channel-window-max",
                type=IntListType(),
                default=None,
                help=(
                    "Comma-separated ints for window.max per channel. Only "
                    "used when any window value is provided."
                ),
            ),
            click.option(
                "--channel-window-start",
                type=IntListType(),
                default=None,
                help=(
                    "Comma-separated ints for window.start per channel. Only "
                    "used when any window value is provided."
                ),
            ),
            click.option(
                "--channel-window-end",
                type=IntListType(),
                default=None,
                help=(
                    "Comma-separated ints for window.end per channel. Only "
                    "used when any window value is provided."
                ),
            ),
            # ── Data / metadata ───────────────────────────────────────────
            click.option(
                "--zarr-format",
                type=click.Choice(["2", "3"], case_sensitive=False),
                default=None,
                help=(
                    "Target Zarr format: '2' ≈ NGFF 0.4, '3' ≈ NGFF 0.5. "
                    "If omitted, use the writer's default."
                ),
            ),
            click.option(
                "--physical-pixel-sizes",
                type=FloatListType(),
                default=None,
                help=(
                    "Comma-separated physical pixel sizes per axis, in the "
                    "same order as --axes-names. Example (t,c,z,y,x): "
                    "'1.0,1.0,0.3,0.108,0.108'. If omitted, use pixel sizes "
                    "from the source metadata when available."
                ),
            ),
            click.option(
                "--dtype",
                default=None,
                help=(
                    "Override output dtype (e.g. 'uint8', 'uint16', 'float32'). "
                    "If omitted, the reader's native dtype is used."
                ),
            ),
            # ── Chunking / sharding (advanced) ────────────────────────────
            click.option(
                "--memory-target",
                type=int,
                default=None,
                help=(
                    "Advanced: approximate in-memory byte budget to derive "
                    "per-level chunk shapes. If set, the writer computes "
                    "chunk shapes from this target (unless explicit "
                    "--chunk-shape/--chunk-shape-per-level are given). "
                    "Defaults to 16 MiB if unset."
                ),
            ),
            click.option(
                "--chunk-shape",
                type=IntListType(),
                default=None,
                help=(
                    "Advanced: single chunk shape tuple, one int per axis. "
                    "Example: '1,1,16,256,256'. Applies to all pyramid levels "
                    "unless --chunk-shape-per-level is provided."
                ),
            ),
            click.option(
                "--chunk-shape-per-level",
                type=IntTupleListType(),
                default=None,
                help=(
                    "Advanced: per-level chunk shapes, level 0 first. "
                    "Semicolon-separated int tuples, each matching the number "
                    "of axes. Example: '1,1,16,256,256;1,1,16,128,128'. "
                    "Overrides --chunk-shape and --memory-target."
                ),
            ),
            click.option(
                "--shard-shape",
                type=IntListType(),
                default=None,
                help=(
                    "Advanced (Zarr v3): single shard shape tuple, one int per "
                    "axis. Example: '1,1,128,1024,1024'. Applies to all "
                    "levels unless --shard-shape-per-level is provided."
                ),
            ),
            click.option(
                "--shard-shape-per-level",
                type=IntTupleListType(),
                default=None,
                help=(
                    "Advanced (Zarr v3): per-level shard shapes, level 0 first. "
                    "Semicolon-separated int tuples, e.g. "
                    "'1,1,128,1024,1024;1,1,128,512,512'. Overrides "
                    "--shard-shape."
                ),
            ),
            # ── Pyramid / multiscale ──────────────────────────────────────
            click.option(
                "--num-levels",
                type=int,
                default=None,
                help=(
                    "Total number of multiscale levels (>=1). If set (and "
                    "--level-shapes is not provided), the writer builds a half "
                    "pyramid in X/Y (and optionally Z with --downsample-z)."
                ),
            ),
            click.option(
                "--downsample-z",
                is_flag=True,
                default=False,
                help=(
                    "With --num-levels, also half the Z dimension at each "
                    "pyramid level when a Z axis exists."
                ),
            ),
            click.option(
                "--level-shapes",
                type=IntTupleListType(),
                default=None,
                help=(
                    "Semicolon-separated per-level SHAPES (ints), level 0 "
                    "first. Each tuple length must match the number of axes. "
                    "Example: '2,3,5,512,512;2,3,5,256,256;2,3,5,128,128'. If "
                    "provided, overrides --num-levels and --downsample-z."
                ),
            ),
            # ── Provenance ────────────────────────────────────────────────
            click.option(
                "--include-provenance",
                is_flag=True,
                default=False,
                help=(
                    "Record source provenance in each store: a top-level "
                    "'bioio_conversion' attributes block naming the source "
                    "file, reader plugin, package versions and conversion "
                    "time, plus the source's standard, OME and native "
                    "metadata as JSON sidecars. Off by default."
                ),
            ),
            click.option(
                "--provenance-reader-kwargs",
                type=JsonDictType(),
                default=None,
                help=(
                    "JSON object of extra kwargs for the provenance metadata "
                    'reader. Example: \'{"plate": "96"}\'. Requires '
                    "--include-provenance."
                ),
            ),
            # ── Job slicing / scene selection ─────────────────────────────
            click.option(
                "--tbatch",
                type=int,
                default=None,
                help=(
                    "Number of timepoints per write batch. Smaller batches "
                    "reduce memory usage at the cost of more I/O overhead."
                ),
            ),
            click.option(
                "--start-t-src",
                type=int,
                default=None,
                help=(
                    "Source T index at which to begin reading (0-based). Maps "
                    "to writer.start_t_src."
                ),
            ),
            click.option(
                "--start-t-dest",
                type=int,
                default=None,
                help=(
                    "Destination T index at which to begin writing (0-based). "
                    "Maps to writer.start_t_dest."
                ),
            ),
            click.option(
                "--scenes",
                "-s",
                type=ScenesType(),
                default=None,
                help=(
                    "Which scene(s) to export, e.g. '0' or '0,2'. Default: all "
                    "scenes in the source."
                ),
            ),
            # ── Locational ─────────────────────────────
            click.option(
                "--name",
                "-n",
                default=None,
                help=(
                    "Base name for output stores (e.g. the group name inside "
                    "the destination directory). If omitted, a default is "
                    "derived from the input or CSV row."
                ),
            ),
            click.option(
                "--destination",
                "-d",
                required=require_destination,
                type=click.Path(),
                help=(
                    "Output directory in which OME-Zarr stores will be created. "
                    "In batch mode, all jobs write under this directory."
                ),
            ),
        ]

        for opt in reversed(opts):
            fn = cast(F, opt(fn))
        return fn

    return _decorator


def build_ome_zarr_init_opts(**kwargs: Any) -> OmeZarrInitOptions:
    """
    Convert Click parameters into `OmeZarrConverter(...)` init kwargs.
    """
    init_opts: OmeZarrInitOptions = {}

    destination = kwargs.get("destination")
    if destination is not None:
        init_opts["destination"] = destination

    zarr_format = kwargs.get("zarr_format")
    if zarr_format is not None:
        init_opts["zarr_format"] = int(zarr_format)

    # Direct pass-through options (same key in click + converter)
    if kwargs.get("name") is not None:
        init_opts["name"] = kwargs["name"]
    if kwargs.get("scenes") is not None:
        init_opts["scenes"] = kwargs["scenes"]
    if kwargs.get("tbatch") is not None:
        init_opts["tbatch"] = kwargs["tbatch"]
    if kwargs.get("start_t_src") is not None:
        init_opts["start_t_src"] = kwargs["start_t_src"]
    if kwargs.get("start_t_dest") is not None:
        init_opts["start_t_dest"] = kwargs["start_t_dest"]

    # Multiscale: explicit level_shapes overrides derived pyramids
    level_shapes = kwargs.get("level_shapes")
    num_levels = kwargs.get("num_levels")
    downsample_z = kwargs.get("downsample_z", False)

    if level_shapes:
        init_opts["level_shapes"] = level_shapes
    elif num_levels is not None:
        init_opts["num_levels"] = num_levels
        if downsample_z:
            init_opts["downsample_z"] = True

    # Chunk/shard: normalize per-level vs single tuple
    if kwargs.get("chunk_shape_per_level"):
        init_opts["chunk_shape"] = kwargs["chunk_shape_per_level"]
    elif kwargs.get("chunk_shape"):
        init_opts["chunk_shape"] = kwargs["chunk_shape"]

    if kwargs.get("shard_shape_per_level"):
        init_opts["shard_shape"] = kwargs["shard_shape_per_level"]
    elif kwargs.get("shard_shape"):
        init_opts["shard_shape"] = kwargs["shard_shape"]

    if kwargs.get("memory_target") is not None:
        init_opts["memory_target"] = kwargs["memory_target"]
    if kwargs.get("dtype") is not None:
        init_opts["dtype"] = kwargs["dtype"]

    pps = kwargs.get("physical_pixel_sizes")
    if pps:
        init_opts["physical_pixel_size"] = list(pps)

    # Channels: only build when labels are provided
    channel_labels = kwargs.get("channel_labels")
    if channel_labels:
        init_opts["channels"] = build_channels(
            labels=channel_labels,
            colors=kwargs.get("channel_colors"),
            actives=kwargs.get("channel_actives"),
            coefs=kwargs.get("channel_coefficients"),
            families=kwargs.get("channel_families"),
            inverted=kwargs.get("channel_inverted"),
            w_min=kwargs.get("channel_window_min"),
            w_max=kwargs.get("channel_window_max"),
            w_start=kwargs.get("channel_window_start"),
            w_end=kwargs.get("channel_window_end"),
        )

    # Provenance
    if kwargs.get("include_provenance"):
        init_opts["include_provenance"] = True
    if kwargs.get("provenance_reader_kwargs") is not None:
        init_opts["provenance_reader_kwargs"] = kwargs["provenance_reader_kwargs"]

    # Axes
    if kwargs.get("axes_names") is not None:
        init_opts["axes_names"] = kwargs["axes_names"]
    if kwargs.get("axes_types") is not None:
        init_opts["axes_types"] = kwargs["axes_types"]
    if kwargs.get("axes_units") is not None:
        init_opts["axes_units"] = kwargs["axes_units"]

    return init_opts
