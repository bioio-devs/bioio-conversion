import itertools
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fsspec
import numcodecs
import numpy as np
import psutil
from bioio import BioImage
from bioio_base.dimensions import DEFAULT_DIMENSION_ORDER, DimensionNames
from bioio_base.reader import Reader
from bioio_ome_zarr.writers import Channel, OMEZarrWriter
from bioio_ome_zarr.writers.ome_zarr_writer import MultiResolutionShapeSpec
from bioio_ome_zarr.writers.utils import multiscale_chunk_size_from_memory_target
from zarr.codecs import BloscCodec

from ..provenance import ProvenanceBuilder, write_sidecars
from ..sharding import (
    DEFAULT_CHUNK_LIMIT_BYTES,
    DEFAULT_SHARD_LIMIT_BYTES,
    build_pyramid_shapes,
    choose_pyramid_layout,
)

# Bounds are ((lo, hi), ...) per axis — a picklable description of a shard region.
_Bounds = Tuple[Tuple[int, int], ...]

DEFAULT_ZARR_FORMAT = 3


def _available_cores() -> int:
    """Number of cores the current process may actually run on.

    Prefer the process's CPU affinity, which reflects the cores granted by a
    cgroup/cpuset (e.g. a SLURM ``--cpus-per-task`` allocation) rather than the
    node's total hardware. ``cpu_affinity`` is unavailable on platforms without
    an affinity API (notably macOS), where it raises ``AttributeError``; there
    we fall back to the physical core count.
    """
    try:
        return max(1, len(psutil.Process().cpu_affinity()))
    except AttributeError:
        return max(1, psutil.cpu_count(logical=False) or 1)


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
    shard_data = np.asarray(
        bio.reader.get_image_data(native_order, **region_kwargs), dtype=out_dtype
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
        n_workers: Optional[int] = None,
        shard_limit_bytes: int = DEFAULT_SHARD_LIMIT_BYTES,
        include_provenance: bool = False,
        provenance_reader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize an OME-Zarr converter with flexible scene selection,
        pyramid construction, and chunk-sizing.

        Parameters
        ----------
        source : str
            Path to the input image (any format supported by BioImage).
        destination : Optional[str]
            Local directory or remote URI under which to write the ``.ome.zarr``
            output(s). If ``None``, the converter will use the current working
            directory
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
            Explicit chunk shape for the written arrays (applies to both Zarr v2
            and v3) — a single tuple applied to all levels, or per-level tuples,
            in the array's axis order (e.g. ``(1, 1, 1, 512, 512)`` for TCZYX).
            The writer validates it against each level's shape. When provided
            under v3 it also disables the auto-shard calculation, since the
            shard layout is derived from the chunk shape.
        shard_shape : Optional[Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]]
            Explicit shard shape (Zarr v3 only).  When provided, disables the
            v3 auto-shard calculation.
        compressor : Optional[Union[zarr.codecs.BloscCodec, numcodecs.abc.Codec]]
            Compression codec. For v2 use ``numcodecs.Blosc``; for v3 use
            ``zarr.codecs.BloscCodec``.
        zarr_format : Optional[int]
            Target Zarr array format (``2`` or ``3``). Defaults to ``3`` when
            ``None``.
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
            Has no effect when ``chunk_shape`` is set explicitly. Default: 16 MiB.
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
        n_workers : Optional[int]
            Number of worker processes for shard writes (auto-layout path).
            Defaults to the number of CPU cores available to the process.
        shard_limit_bytes : int
            Maximum uncompressed size of a level-0 shard. Default: 4 GiB.
        include_provenance : bool, default = False
            When True, write source provenance for each scene into a top-level
            ``"bioio_conversion"`` attributes block: the source's ``standard_metadata``,
            the reader plugin and package versions, and the native/OME metadata
            as JSON sidecars under ``bioio/``. Off by default; see
            :class:`bioio_conversion.provenance.ProvenanceBuilder`.
        provenance_reader_kwargs : dict, optional
            Extra kwargs forwarded to ``BioImage`` when opening a dedicated
            metadata reader for provenance. When ``None`` (default) the pixel
            reader is reused as-is. Ignored when ``include_provenance=False``.
        """
        self.source = source
        self.destination = destination or str(Path.cwd())
        self.output_basename = name or Path(source).stem

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
        self._writer_zarr_format = (
            DEFAULT_ZARR_FORMAT if zarr_format is None else zarr_format
        )
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
        # Default to one process per available core (shard writes are GIL-bound
        # CPU work). _available_cores honors a cgroup/SLURM allocation so we do
        # not oversubscribe a partial node; it is floored at 1.
        self._n_workers = n_workers or _available_cores()
        self._shard_limit_bytes = shard_limit_bytes
        # Provenance (the "bioio_conversion" attribute block + source-metadata sidecars)
        self._provenance = (
            ProvenanceBuilder(
                self.source,
                self.bioimage,
                self.scene_names,
                metadata_reader_kwargs=provenance_reader_kwargs,
            )
            if include_provenance
            else None
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _infer_physical_pixel_sizes(
        self, axis_names: List[str]
    ) -> Optional[List[float]]:
        """Per-axis level-0 scale for the writer.

        Spatial axes (Z, Y, X) come from ``BioImage.scale`` (physical pixel
        sizes). The time axis (T) is the acquisition **time interval** in
        seconds.
        """
        if self._writer_physical_pixel_size is not None:
            return [float(x) for x in self._writer_physical_pixel_size]

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

    def _infer_axes_units(self, axis_names: List[str]) -> Optional[List[Optional[str]]]:
        # Override.
        if self._writer_axes_units is not None:
            return self._writer_axes_units

        # Otherwise fetch from BioImage
        dim_props = getattr(self.bioimage, "dimension_properties", None)
        if dim_props is None:
            return None

        mapping = {
            "t": getattr(dim_props, "T", None),
            "c": getattr(dim_props, "C", None),
            "z": getattr(dim_props, "Z", None),
            "y": getattr(dim_props, "Y", None),
            "x": getattr(dim_props, "X", None),
        }
        units: List[Optional[str]] = []
        for ax in axis_names:
            prop = mapping.get(ax)
            unit = getattr(prop, "unit", None) if prop is not None else None
            units.append(str(unit) if unit is not None else None)

        # Fallback = None
        if all(unit is None for unit in units):
            return None
        return units

    def _resolve_channels(
        self, axis_names: List[str], channel_count: int
    ) -> Optional[List[Channel]]:
        """
        Resolve channel metadata for the writer.

        Policy:
        - If the user explicitly provided channels, always honor them
        (even if no 'c' axis is present).
        - Otherwise, only derive channels if a 'c' axis exists.
        """

        # 1. User explicitly supplied channels → always use them
        if self._writer_channels is not None:
            return self._writer_channels

        # 2. No channel axis → no channels to derive
        if "c" not in axis_names:
            return None

        # 3. Derive minimal channels from BioImage metadata
        labels = self.bioimage.channel_names or [
            f"Channel:{i}" for i in range(channel_count)
        ]

        return [Channel(label=lab, color="#FFFFFF") for lab in labels[:channel_count]]

    def _native_axes_and_shape_for_scene(
        self, scene_index: int
    ) -> Tuple[List[str], Tuple[int, ...]]:
        """
        Use BioImage.reader (the actual format plugin) to discover true
        axis order & shape. This reflects CYX, CZYX, TCZYX, etc.
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

    def _build_pyramid_shapes_simple(
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

        fs, _ = fsspec.core.url_to_fs(self.destination)
        is_local = fsspec.utils.get_protocol(self.destination) == "file"
        out_paths: Dict[int, str] = {}
        for idx in self.scene_indices:
            base = self._output_base_for_scene(idx)
            if is_local:
                out_paths[idx] = str(Path(self.destination) / f"{base}.ome.zarr")
            else:
                out_paths[idx] = self.destination.rstrip("/") + f"/{base}.ome.zarr"
        for path in out_paths.values():
            if fs.exists(path):
                raise FileExistsError(f"{path} already exists.")

        # A single process pool spans *all* scenes
        pool = ProcessPoolExecutor(max_workers=self._n_workers)
        futures: List[Any] = []
        try:
            for scene_index in self.scene_indices:
                futures.extend(
                    self._plan_and_dispatch_scene(
                        scene_index, out_paths[scene_index], pool
                    )
                )
            for future in futures:
                future.result()
        finally:
            pool.shutdown()

    def _output_base_for_scene(self, scene_index: int) -> str:
        """Sanitized output basename (no extension) for a scene's store.

        Single-scene conversions use the base name as-is; multi-scene runs
        suffix each store with the scene name.
        """
        scene_name = self.scene_names[scene_index]
        basename = (
            self.output_basename
            if len(self.scene_indices) == 1
            else f"{self.output_basename}_{scene_name}"
        )
        return re.sub(r"[<>:\"/\\|?*]", "_", basename)

    def _plan_and_dispatch_scene(
        self,
        scene_index: int,
        out_path: str,
        pool: ProcessPoolExecutor,
    ) -> List[Any]:
        """
        Build one scene's store, initialize it, and dispatch its writes.

        Returns the list of futures submitted to ``pool`` (empty when the
        fallback single-threaded path is used).
        """
        bio = self.bioimage
        base = self._output_base_for_scene(scene_index)

        # (1) Discover native axes/shape from the active reader
        axis_names, level0_shape = self._native_axes_and_shape_for_scene(scene_index)

        # (2) Channels
        r = bio.reader
        ccount = int(getattr(r.dims, "C", 1)) if "c" in axis_names else 0
        channels = self._resolve_channels(axis_names, ccount)
        pps = self._infer_physical_pixel_sizes(axis_names)

        dims = "".join(ax.upper() for ax in axis_names)

        # True when Y and X are present and all dims fit within TCZYX.ß
        ome_dims = (
            {
                DimensionNames.SpatialY,
                DimensionNames.SpatialX,
            }
            <= set(dims)
            <= set(DEFAULT_DIMENSION_ORDER)
        )

        # (3) Scale to writer
        if self._writer_level_shapes is not None:
            writer_level_shapes_param: MultiResolutionShapeSpec = (
                self._writer_level_shapes
            )
        elif (
            self._writer_zarr_format == 3
            and self._helper_num_levels is None
            and ome_dims
        ):
            # v3 default: auto-generate pyramid levels down to atlas fit.
            writer_level_shapes_param = build_pyramid_shapes(level0_shape, dims)
        else:
            derived = self._build_pyramid_shapes_simple(axis_names, level0_shape)
            writer_level_shapes_param = (
                derived if derived is not None else tuple(level0_shape)
            )

        # (4) Chunking + sharding
        can_auto_layout = (
            self._writer_zarr_format == 3
            and self._writer_chunk_shape is None
            and self._writer_shard_shape is None
            and ome_dims
        )
        (
            writer_chunk_shape_param,
            writer_shard_shape_param,
        ) = self._resolve_chunk_and_shard_params(
            can_auto_layout, writer_level_shapes_param, dims
        )

        # (5) Build writer kwargs
        if self._provenance is not None:
            bioio_attrs, bioio_sidecars = self._provenance.provenance_from_scene(
                scene_index
            )
        else:
            bioio_attrs, bioio_sidecars = None, {}
        writer_kwargs: Dict[str, Any] = {
            "store": out_path,
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
                    "channels": channels,
                    "rdefs": self._writer_rdefs,
                    "creator_info": self._writer_creator_info,
                    "root_transform": self._writer_root_transform,
                    "axes_names": (self._writer_axes_names or axis_names),
                    "axes_types": self._writer_axes_types,
                    "axes_units": self._infer_axes_units(axis_names),
                    "physical_pixel_size": pps,
                    "attributes": bioio_attrs,
                }.items()
                if v is not None
            },
        }

        writer = OMEZarrWriter(**writer_kwargs)

        # (6) Read pixels from the reader in its native axis order
        bio.set_scene(scene_index)
        r = bio.reader
        native_order = r.dims.order.upper()

        # (7) Write — each call covers exactly one shard boundary at every
        # pyramid level, so shards can be written in any order independently.
        writer.initialize()
        write_sidecars(Path(out_path), bioio_sidecars)

        scene_futures: List[Any] = []
        if can_auto_layout:
            out_dtype_str = str(self.output_dtype)
            for src_bounds, dest_bounds in self._scene_shard_bounds(
                writer_shard_shape_param, level0_shape
            ):
                task = (
                    self.source,
                    str(out_path),
                    native_order,
                    scene_index,
                    out_dtype_str,
                    src_bounds,
                    dest_bounds,
                )
                scene_futures.append(pool.submit(_write_shard_process, *task))
        else:
            t_ax = dims.index("T") if "t" in axis_names else None
            self._write_fallback(writer, r, native_order, t_ax, level0_shape)
        return scene_futures

    def _resolve_chunk_and_shard_params(
        self,
        can_auto_layout: bool,
        writer_level_shapes_param: MultiResolutionShapeSpec,
        dims: str,
    ) -> Tuple[Optional[MultiResolutionShapeSpec], Optional[MultiResolutionShapeSpec]]:
        """Return ``(chunk_shape_param, shard_shape_param)`` for the writer.

        Prefers an explicit user-supplied chunk shape; falls back to auto-layout
        via ``choose_pyramid_layout``, then memory-target heuristics, then
        ``None`` (writer default).
        """
        shard_param = self._writer_shard_shape
        if self._writer_chunk_shape is not None:
            return self._writer_chunk_shape, shard_param
        elif can_auto_layout:
            chunk_limit = self._helper_memory_target_bytes or DEFAULT_CHUNK_LIMIT_BYTES
            level_shapes_list = self._ensure_per_level_shapes(writer_level_shapes_param)
            auto_chunks, auto_shards = choose_pyramid_layout(
                level_shapes=level_shapes_list,
                dtype=self.output_dtype,
                dims=dims,
                chunk_limit_bytes=chunk_limit,
                shard_limit_bytes=self._shard_limit_bytes,
            )
            return auto_chunks, auto_shards
        elif self._helper_memory_target_bytes is not None:
            level_shapes_list = self._ensure_per_level_shapes(writer_level_shapes_param)
            suggested = multiscale_chunk_size_from_memory_target(
                level_shapes_list,
                self.output_dtype,
                self._helper_memory_target_bytes,
            )
            return [tuple(map(int, s)) for s in suggested], shard_param
        else:
            return None, shard_param  # writer suggests per-level ~16 MiB

    def _scene_shard_bounds(
        self,
        auto_shards: MultiResolutionShapeSpec,
        level0_shape: Tuple[int, ...],
    ) -> List[Tuple[_Bounds, _Bounds]]:
        """Enumerate a scene's level-0 shard regions, one entry per shard.

        Returns ``(src_bounds, dest_bounds)`` pairs tiling ``level0_shape`` into
        disjoint, shard-aligned boxes. Each region maps to exactly one shard, so
        it can be written independently and in any order.
        """
        shard0 = auto_shards[0]
        per_ax = [
            [
                (s, min(s + shard0[ax], level0_shape[ax]))
                for s in range(0, level0_shape[ax], shard0[ax])
            ]
            for ax in range(len(level0_shape))
        ]
        return [(bounds, bounds) for bounds in itertools.product(*per_ax)]

    def _write_fallback(
        self,
        writer: OMEZarrWriter,
        reader: Reader,
        native_order: str,
        t_ax: Optional[int],
        level0_shape: Tuple[int, ...],
    ) -> None:
        """Single-threaded fallback write for non-auto-layout stores.

        Reads and writes the image in ``self._tbatch``-sized batches along T,
        defaulting to one timepoint per write so peak memory stays bounded.  The
        per-batch T slice is delegated to the reader's dimension kwargs (lazily,
        via ``get_image_dask_data``).  When there is no T axis the image is a
        single volume, written as one batch.
        """
        # Source/destination may start at different T offsets (e.g. appending to
        # an existing store), so track the read and write T origins separately.
        base_t_src = self._start_t_src or 0
        base_t_dest = self._start_t_dest or 0

        # No T axis → treat the whole volume as a single batch (one iteration).
        t_total = level0_shape[t_ax] if t_ax is not None else 1
        batch_size = self._tbatch or 1

        for i in range(0, t_total, batch_size):
            t_end = min(i + batch_size, t_total)

            # Default to the full extent on every axis; only T is sub-sliced.
            dest_slices: List[slice] = [slice(0, s) for s in level0_shape]
            read_kwargs: Dict[str, slice] = {}
            if t_ax is not None:
                # Read this T window from the source; write it at the (possibly
                # offset) destination T window. Other axes stay full-extent.
                read_kwargs[DimensionNames.Time] = slice(
                    base_t_src + i, base_t_src + t_end
                )
                dest_slices[t_ax] = slice(base_t_dest + i, base_t_dest + t_end)

            # get_image_dask_data slices lazily, so .compute() only materializes
            # this batch — not the whole image.
            batch = reader.get_image_dask_data(native_order, **read_kwargs)
            writer.write_region(batch.compute(), tuple(dest_slices))
