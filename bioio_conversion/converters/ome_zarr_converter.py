import itertools
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numcodecs
import numpy as np
from bioio import BioImage
from bioio_base.dimensions import DimensionNames
from bioio_ome_zarr.writers import Channel, OMEZarrWriter
from bioio_ome_zarr.writers.ome_zarr_writer import MultiResolutionShapeSpec
from bioio_ome_zarr.writers.utils import multiscale_chunk_size_from_memory_target
from zarr.codecs import BloscCodec

from ..channel_colors import get_channel_colors
from ..cluster import Cluster
from ..sharding import (
    DEFAULT_CHUNK_LIMIT_BYTES,
    DEFAULT_SHARD_LIMIT_BYTES,
    _build_pyramid_shapes,
    _choose_pyramid_layout,
)

# Bounds are ((lo, hi), ...) per axis — a picklable description of a shard region.
_Bounds = Tuple[Tuple[int, int], ...]


def _write_shard_process(
    source: str,
    store_path: str,
    native_order: str,
    scene_index: int,
    out_dtype_str: str,
    src_bounds: _Bounds,
    dest_bounds: _Bounds,
) -> None:
    """
    Read one shard from the source and write it to an already-initialized
    OME-Zarr store. Module-level and primitive-only (so its arguments pickle
    cleanly) to run in a separate process.

    The write phase (downsample + Blosc compression + zarr shard assembly) is
    largely GIL-bound, so process-based parallelism — not threads — is what
    actually scales it. Workers attach to the existing store and only call
    ``write_region``; the store is created once by the parent before dispatch,
    and every worker writes a disjoint shard, so there is no coordination or
    read-modify-write between processes.
    """
    src_region = tuple(slice(lo, hi) for lo, hi in src_bounds)
    dest_region = tuple(slice(lo, hi) for lo, hi in dest_bounds)
    out_dtype = np.dtype(out_dtype_str)

    # New image instance to access data per process
    bio = BioImage(source)
    bio.set_scene(scene_index)
    region_kwargs = {
        native_order[i]: slice(src_region[i].start, src_region[i].stop)
        for i in range(len(native_order))
    }
    # Read the shard via the reader's get_image_data slicing.
    shard_data = bio.reader.get_image_data(native_order, **region_kwargs).astype(
        out_dtype, copy=False
    )

    # Attach to the store the parent already initialized and write this shard.
    writer = OMEZarrWriter.open(store_path)
    writer.write_region(shard_data, dest_region)


class OmeZarrConverter:
    """
    OmeZarrConverter handles conversion of any BioImage‐supported format
    (TIFF, CZI, etc.) into OME-Zarr stores. Supports exporting one, many, or
    all scenes from a multi-scene file.
    """

    def __init__(
        self,
        *,
        source: str,
        destination: Optional[str] = None,
        scenes: Optional[Union[int, List[int]]] = None,
        name: Optional[str] = None,
        level_shapes: Optional[MultiResolutionShapeSpec] = None,
        chunk_shape: Optional[MultiResolutionShapeSpec] = None,
        shard_shape: Optional[MultiResolutionShapeSpec] = None,
        compressor: Optional[Union[BloscCodec, numcodecs.abc.Codec]] = None,
        zarr_format: Optional[int] = None,
        image_name: Optional[str] = None,
        channels: Optional[List[Channel]] = None,
        rdefs: Optional[Dict[str, Any]] = None,
        creator_info: Optional[Dict[str, Any]] = None,
        root_transform: Optional[Dict[str, Any]] = None,
        axes_names: Optional[List[str]] = None,
        axes_types: Optional[List[str]] = None,
        axes_units: Optional[List[Optional[str]]] = None,
        physical_pixel_size: Optional[List[float]] = None,
        num_levels: Optional[int] = None,
        downsample_z: bool = False,
        memory_target: Optional[int] = None,
        start_t_src: Optional[int] = None,
        start_t_dest: Optional[int] = None,
        tbatch: Optional[int] = None,
        dtype: Optional[Union[str, np.dtype]] = None,
        auto_dask_cluster: bool = False,
        n_workers: int = 1,
        shard_limit_bytes: int = DEFAULT_SHARD_LIMIT_BYTES,
    ) -> None:
        """
        Initialize an OME-Zarr converter with flexible scene selection,
        pyramid construction, and chunk-sizing.

        Parameters
        ----------
        source : str
            Path to the input image (any format supported by BioImage).
        destination : Optional[str]
            Directory in which to write the ``.ome.zarr`` output(s).
            If ``None``, the converter will use the current working directory
        scenes : Optional[Union[int, List[int]]]
            Which scene(s) to export:
            - ``None`` → export all scenes
            - ``int``  → a single scene index
            - ``List[int]`` → those specific scene indices
        name : Optional[str]
            Base name for output files (defaults to the source stem). When exporting
            multiple scenes, each file name is suffixed with the scene’s name.
        level_shapes : Optional[List[Tuple[int, ...]]]
            Explicit per-level, per-axis absolute shapes (level 0 first).
            Each tuple length must match the native axis count.
            If provided, convenience options like ``num_levels`` and ``downsample_z``
            are ignored.
        chunk_shape : Optional[Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]]
            Explicit chunk shape — a single tuple applied to all levels or
            per-level tuples.  When provided, disables the v3 auto-chunk/shard
            calculation.
        shard_shape : Optional[Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]]
            Explicit shard shape (Zarr v3 only).  When provided, disables the
            v3 auto-shard calculation.
        compressor : Optional[Union[zarr.codecs.BloscCodec, numcodecs.abc.Codec]]
            Compression codec. For v2 use ``numcodecs.Blosc``; for v3 use
            ``zarr.codecs.BloscCodec``.
        zarr_format : Optional[int]
            Target Zarr array format (``2`` or ``3``). ``None`` lets the writer
            choose its default.
        image_name : Optional[str]
            Image name to record in multiscales metadata. Defaults to the output base.
        channels : Optional[List[Channel]]
            Optional OMERO-style channel metadata. Only used when a ``'c'`` axis
            exists. If omitted, minimal channel models are derived from the reader.
        rdefs : Optional[Dict[str, Any]]
            Optional OMERO rendering defaults.
        creator_info : Optional[Dict[str, Any]]
            Optional “creator” metadata block (e.g., tool/version).
        root_transform : Optional[Dict[str, Any]]
            Optional multiscale root coordinate transformation.
        axes_names : Optional[List[str]]
            Axis names to write; defaults to the native axis names from the reader.
        axes_types : Optional[List[str]]
            Axis types (e.g., ``["time","channel","space",...]``). Writer validates.
        axes_units : Optional[List[Optional[str]]]
            Physical units per axis. Writer validates.
        physical_pixel_size : Optional[List[float]]
            Physical scale at level 0 per axis. If omitted, values are derived from
            ``BioImage.scale`` for present axes.
        num_levels : Optional[int]
            Number of pyramid levels (including level 0) to generate via a
            simple XY half-pyramid.  When set, overrides the v3 atlas-based
            auto-pyramid:

            - ``1`` = only level 0
            - ``2`` = level 0 + one XY half
            - ``3`` = level 0 + two XY halves, etc.

            Ignored if ``level_shapes`` is provided.
        downsample_z : bool, default = False
            Also halve Z at each level when building the ``num_levels`` pyramid.
            Ignored if ``level_shapes`` is provided.
        memory_target : Optional[int]
            Chunk budget in bytes.  For ``zarr_format=3`` this is passed as
            ``chunk_limit_bytes`` to the auto-chunk/shard layout; for other
            formats it drives ``multiscale_chunk_size_from_memory_target``.
            Has no effect when ``chunk_shape`` is set explicitly.
        start_t_src : Optional[int]
            Source T index at which to begin reading from the BioImage. Default: use
            writer default.
        start_t_dest : Optional[int]
            Destination T index at which to begin writing into the store. Default:
            use writer default.
        tbatch : Optional[int]
            Number of timepoints to transfer. If None, the converter writes as many
            as available in both source and destination.
        dtype : Optional[Union[str, np.dtype]]
            Override output data type; defaults to the reader’s dtype.
        auto_dask_cluster : bool
            If True, automatically spin up a local Dask cluster with
            8 workers (using `Cluster(n_workers=8).start()`) before any
            conversion.
        n_workers : int
            Number of worker *processes* for shard writes (auto-layout path).
        shard_limit_bytes : int
            Maximum uncompressed size of a level-0 shard. Default: 4 GiB.
        """
        self.source = source
        self.destination = destination or str(Path.cwd())
        self.output_basename = name or Path(source).stem

        # Optional local Dask cluster
        if auto_dask_cluster:
            cluster = Cluster(n_workers=8)
            cluster.start()

        self.bioimage = BioImage(self.source)
        self.scene_names = self.bioimage.scenes
        nscenes = len(self.scene_names)

        if scenes is None:
            self.scene_indices = list(range(nscenes))
        elif isinstance(scenes, int):
            self.scene_indices = [scenes]
        else:
            self.scene_indices = list(scenes)

        self.bioimage.set_scene(0)
        self.output_dtype = (
            np.dtype(dtype) if dtype is not None else self.bioimage.dtype
        )

        # Passthroughs
        self._writer_level_shapes = level_shapes
        self._writer_chunk_shape = chunk_shape
        self._writer_shard_shape = shard_shape
        self._writer_compressor = compressor
        self._writer_zarr_format = zarr_format
        self._writer_image_name = image_name
        self._writer_channels = channels
        self._writer_rdefs = rdefs
        self._writer_creator_info = creator_info
        self._writer_root_transform = root_transform
        self._writer_axes_names = axes_names
        self._writer_axes_types = axes_types
        self._writer_axes_units = axes_units
        self._writer_physical_pixel_size = physical_pixel_size

        # Helpers
        self._helper_num_levels = num_levels
        self._helper_downsample_z = downsample_z

        # Chunk suggestion
        self._helper_memory_target_bytes = (
            None if memory_target is None else memory_target
        )
        self._start_t_src = start_t_src
        self._start_t_dest = start_t_dest
        self._tbatch = None if tbatch is None else tbatch
        self._n_workers = n_workers
        self._shard_limit_bytes = shard_limit_bytes

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _infer_physical_pixel_sizes(
        self, axis_names: List[str]
    ) -> Optional[List[float]]:
        if self._writer_physical_pixel_size is not None:
            return [float(x) for x in self._writer_physical_pixel_size]

        # From BioImage.scale; include only present axes
        scale_info = self.bioimage.scale
        defaults = {"t": 1.0, "z": 1.0, "y": 1.0, "x": 1.0, "c": 1.0}
        mapping = {
            "t": getattr(scale_info, "T", None),
            "z": getattr(scale_info, "Z", None),
            "y": getattr(scale_info, "Y", None),
            "x": getattr(scale_info, "X", None),
            "c": 1.0,
        }
        return [
            float(mapping.get(ax, defaults[ax]) or defaults[ax]) for ax in axis_names
        ]

    def _resolve_channels(
        self, axis_names: List[str], channel_count: int
    ) -> Optional[List[Channel]]:
        """
        Resolve channel metadata for the writer.

        Policy:
        - If the user explicitly provided channels, always honor them
        (even if no 'c' axis is present).
        - Otherwise, only derive channels if a 'c' axis exists, auto-assigning
        display colors via :func:`get_channel_colors` (brightfield → white,
        fluorescent channels → a perceptually-distinct palette).
        """

        # 1. User explicitly supplied channels → always use them
        if self._writer_channels is not None:
            return self._writer_channels

        # 2. No channel axis → no channels to derive
        if "c" not in axis_names:
            return None

        # 3. Derive minimal channels from BioImage metadata, auto-detecting a
        #    display color per channel. ``self.bioimage`` is already set to the
        #    active scene by the caller, so get_channel_colors inspects it.
        labels = self.bioimage.channel_names or [
            f"Channel:{i}" for i in range(channel_count)
        ]
        colors = get_channel_colors(self.bioimage)

        return [
            Channel(
                label=lab,
                color=colors[i] if i < len(colors) else "FFFFFF",
            )
            for i, lab in enumerate(labels[:channel_count])
        ]

    def _native_axes_and_shape_for_scene(
        self, scene_index: int
    ) -> Tuple[List[str], Tuple[int, ...]]:
        """
        Use BioImage.reader (the actual format plugin) to discover true
        axis order & shape. This reflects CYX, CZYX, TCZYX, etc., without
        padding.
        """
        self.bioimage.set_scene(scene_index)
        r = self.bioimage.reader
        order = r.dims.order.upper()
        axis_names = [c.lower() for c in order]
        shape = tuple(int(getattr(r.dims, ax)) for ax in order)
        return axis_names, shape

    def _round_shape(
        self, base_shape: Tuple[int, ...], factors: Tuple[float, ...]
    ) -> Tuple[int, ...]:
        """
        Apply per-axis factors to `base_shape`; clamp each dim to >= 1.
        """
        return tuple(max(1, int(round(d * f))) for d, f in zip(base_shape, factors))

    def _build_level_shapes_simple(
        self,
        axis_names: List[str],
        level0_shape: Tuple[int, ...],
    ) -> Optional[List[Tuple[int, ...]]]:
        """
        Build per-level shapes from (num_levels, downsample_z) policy.

        - If num_levels <= 1 or None → return None (single level).
        - Else produce half-pyramid:
            * XY always downsample by 0.5^level.
            * If downsample_z=True and 'z' exists, Z also downsample by 0.5^level.
            * t/c/other axes remain unchanged.
        """
        if not self._helper_num_levels or self._helper_num_levels <= 1:
            return None

        result: List[Tuple[int, ...]] = [tuple(level0_shape)]
        for lvl in range(1, self._helper_num_levels):
            factors: List[float] = []
            for ax in axis_names:
                if ax in ("x", "y"):
                    factors.append(0.5**lvl)
                elif ax == "z" and self._helper_downsample_z:
                    factors.append(0.5**lvl)
                else:
                    factors.append(1.0)
            result.append(self._round_shape(level0_shape, tuple(factors)))
        return result

    @staticmethod
    def _ensure_per_level_shapes(
        level_shapes_spec: MultiResolutionShapeSpec,
    ) -> List[Tuple[int, ...]]:
        """
        Normalize a level-shape spec (single or per-level) into a per-level
        list of tuples.
        """
        if len(level_shapes_spec) == 0:
            raise ValueError("level_shapes cannot be empty")
        first = level_shapes_spec[0]
        if isinstance(first, (int, np.integer)):
            # Single level-0 shape
            return [tuple(int(x) for x in level_shapes_spec)]
        # Already per-level
        return [tuple(int(x) for x in level) for level in level_shapes_spec]

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------

    def convert(self) -> None:
        if len(self.scene_indices) > 1:
            bad = [
                nm
                for i, nm in enumerate(self.scene_names)
                if i in self.scene_indices and re.search(r"[<>:\"/\\|?*]", nm)
            ]
            if bad:
                warnings.warn(
                    (
                        "Scene names contain invalid characters and will be "
                        "sanitized in filenames: "
                        f"{bad}"
                    ),
                    UserWarning,
                )

        bio = self.bioimage
        for scene_index in self.scene_indices:
            # (1) Discover native axes/shape from the active reader
            axis_names, level0_shape = self._native_axes_and_shape_for_scene(
                scene_index
            )

            # (2) Channels
            r = bio.reader
            ccount = int(getattr(r.dims, "C", 1)) if "c" in axis_names else 0
            channel_models = self._resolve_channels(axis_names, ccount)
            pps = self._infer_physical_pixel_sizes(axis_names)

            dims = "".join(ax.upper() for ax in axis_names)

            _ome_dims = (
                DimensionNames.SpatialY in dims
                and DimensionNames.SpatialX in dims
                and all(
                    d
                    in (
                        DimensionNames.Time,
                        DimensionNames.Channel,
                        DimensionNames.SpatialZ,
                        DimensionNames.SpatialY,
                        DimensionNames.SpatialX,
                    )
                    for d in dims
                )
            )

            # (3) Scale to writer
            if self._writer_level_shapes is not None:
                writer_level_shapes_param: MultiResolutionShapeSpec = (
                    self._writer_level_shapes
                )
            elif (
                self._writer_zarr_format == 3
                and self._helper_num_levels is None
                and _ome_dims
            ):
                # v3 default: auto-generate pyramid levels down to atlas fit.
                writer_level_shapes_param = _build_pyramid_shapes(level0_shape, dims)
            else:
                derived = self._build_level_shapes_simple(axis_names, level0_shape)
                writer_level_shapes_param = (
                    derived if derived is not None else tuple(level0_shape)
                )

            # (4) Chunking + sharding
            writer_shard_shape_param = self._writer_shard_shape

            _can_auto_layout = (
                self._writer_zarr_format == 3
                and self._writer_chunk_shape is None
                and self._writer_shard_shape is None
                and _ome_dims
            )

            if self._writer_chunk_shape is not None:
                writer_chunk_shape_param: Optional[
                    MultiResolutionShapeSpec
                ] = self._writer_chunk_shape
            elif _can_auto_layout:
                chunk_limit = (
                    self._helper_memory_target_bytes or DEFAULT_CHUNK_LIMIT_BYTES
                )
                level_shapes_list = self._ensure_per_level_shapes(
                    writer_level_shapes_param
                )
                auto_chunks, auto_shards = _choose_pyramid_layout(
                    level_shapes=level_shapes_list,
                    dtype=self.output_dtype,
                    dims=dims,
                    chunk_limit_bytes=chunk_limit,
                    shard_limit_bytes=self._shard_limit_bytes,
                )
                writer_chunk_shape_param = auto_chunks
                writer_shard_shape_param = auto_shards
            elif self._helper_memory_target_bytes is not None:
                # Normalize level shapes to per-level list for the helper
                level_shapes_list = self._ensure_per_level_shapes(
                    writer_level_shapes_param
                )
                suggested = multiscale_chunk_size_from_memory_target(
                    level_shapes_list,
                    self.output_dtype,
                    self._helper_memory_target_bytes,
                )
                writer_chunk_shape_param = [tuple(map(int, s)) for s in suggested]
            else:
                writer_chunk_shape_param = None  # writer suggests per-level ~16 MiB

            # (5) Output path
            scene_name = self.scene_names[scene_index]
            base = (
                self.output_basename
                if len(self.scene_indices) == 1
                else f"{self.output_basename}_{scene_name}"
            )
            base = re.sub(r"[<>:\"/\\|?*]", "_", base)
            out_path = Path(self.destination) / f"{base}.ome.zarr"
            if out_path.exists():
                raise FileExistsError(f"{out_path} already exists.")

            # (6) Build writer kwargs
            writer_kwargs: Dict[str, Any] = {
                "store": str(out_path),
                "level_shapes": writer_level_shapes_param,
                "dtype": self.output_dtype,
                **{
                    k: v
                    for k, v in {
                        "chunk_shape": writer_chunk_shape_param,
                        "shard_shape": writer_shard_shape_param,
                        "compressor": self._writer_compressor,
                        "zarr_format": self._writer_zarr_format,
                        "image_name": (self._writer_image_name or base),
                        "channels": channel_models,
                        "rdefs": self._writer_rdefs,
                        "creator_info": self._writer_creator_info,
                        "root_transform": self._writer_root_transform,
                        "axes_names": (self._writer_axes_names or axis_names),
                        "axes_types": self._writer_axes_types,
                        "axes_units": self._writer_axes_units,
                        "physical_pixel_size": pps,
                    }.items()
                    if v is not None
                },
            }

            writer = OMEZarrWriter(**writer_kwargs)

            # (7) Read pixels directly from the reader in its native (unpadded) order
            bio.set_scene(scene_index)
            r = bio.reader
            native_order = r.dims.order.upper()
            data_all = r.get_image_dask_data(native_order)

            # (8) Write — shard-aligned region writes via write_region.
            # Every call covers exactly one shard boundary at every pyramid level
            # so no shard is ever read back to be merged (no read-modify-write).
            writer.initialize()

            if _can_auto_layout:
                self._write_auto_layout_shards(
                    out_path, native_order, scene_index, auto_shards, level0_shape
                )
            else:
                has_t = "t" in axis_names
                t_ax = dims.index("T") if has_t else None
                self._write_fallback(writer, data_all, has_t, t_ax, level0_shape)

    def _write_auto_layout_shards(
        self,
        out_path: Path,
        native_order: str,
        scene_index: int,
        auto_shards: MultiResolutionShapeSpec,
        level0_shape: Tuple[int, ...],
    ) -> None:
        """Write each disjoint level-0 shard via its own ``write_region`` call.

        Every shard covers exactly one shard boundary at every pyramid level, so
        no shard is ever read back to be merged. The write phase is GIL-bound, so
        when ``n_workers > 1`` the shards are dispatched to a
        ``ProcessPoolExecutor`` (the parent has already created the store via
        ``writer.initialize()``); otherwise they are written in-process.
        """
        shard0 = auto_shards[0]
        per_ax = [
            [
                (s, min(s + shard0[ax], level0_shape[ax]))
                for s in range(0, level0_shape[ax], shard0[ax])
            ]
            for ax in range(len(level0_shape))
        ]
        shard_bounds: List[Tuple[_Bounds, _Bounds]] = [
            (bounds, bounds) for bounds in itertools.product(*per_ax)
        ]
        out_dtype_str = str(self.output_dtype)
        n_workers = self._n_workers

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(
                        _write_shard_process,
                        self.source,
                        str(out_path),
                        native_order,
                        scene_index,
                        out_dtype_str,
                        src_bounds,
                        dest_bounds,
                    )
                    for src_bounds, dest_bounds in shard_bounds
                ]
                for future in futures:
                    future.result()  # surface any worker exception
        else:
            for src_bounds, dest_bounds in shard_bounds:
                _write_shard_process(
                    self.source,
                    str(out_path),
                    native_order,
                    scene_index,
                    out_dtype_str,
                    src_bounds,
                    dest_bounds,
                )

    def _write_fallback(
        self,
        writer: OMEZarrWriter,
        data_all: Any,
        has_t: bool,
        t_ax: Optional[int],
        level0_shape: Tuple[int, ...],
    ) -> None:
        """Single-threaded fallback write for non-auto-layout stores.

        Writes the whole image in one region when there is no T axis, otherwise
        in ``self._tbatch``-sized batches along T.
        """
        if not has_t:
            writer.write_region(
                data_all.compute(),
                tuple(slice(0, s) for s in level0_shape),
            )
            return

        assert t_ax is not None
        base_t_src = self._start_t_src or 0
        base_t_dest = self._start_t_dest or 0
        t_total = level0_shape[t_ax]
        batch_size = self._tbatch or t_total
        for i in range(0, t_total, batch_size):
            t_end = min(i + batch_size, t_total)
            dest_slices: List[slice] = [
                slice(0, level0_shape[ax]) for ax in range(len(level0_shape))
            ]
            dest_slices[t_ax] = slice(base_t_dest + i, base_t_dest + t_end)
            src_slices: List[slice] = list(dest_slices)
            src_slices[t_ax] = slice(base_t_src + i, base_t_src + t_end)
            writer.write_region(
                data_all[tuple(src_slices)].compute(),
                tuple(dest_slices),
            )
