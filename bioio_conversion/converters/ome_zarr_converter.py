import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numcodecs
import numpy as np
from bioio import BioImage
from bioio_ome_zarr.writers import Channel, OMEZarrWriter
from bioio_ome_zarr.writers.ome_zarr_writer import ChunkShape
from bioio_ome_zarr.writers.utils import chunk_size_from_memory_target
from zarr.codecs import BloscCodec

from ..cluster import Cluster


class OmeZarrConverter:
    """
    OmeZarrConverter handles conversion of any BioImage‐supported format
    (TIFF, CZI, etc.) into OME-Zarr stores. Supports exporting one,
    many, or all scenes from a multi-scene file, with configurable
    multi-resolution pyramids (full scales or per-axis XY/Z
    factors) and chunk-size memory targeting.
    """

    def __init__(
        self,
        *,
        source: str,
        destination: str,
        scenes: Optional[Union[int, List[int]]] = None,
        name: Optional[str] = None,
        scale: Optional[Tuple[Tuple[float, ...], ...]] = None,
        chunk_shape: Optional[ChunkShape] = None,
        shard_factor: Optional[Tuple[int, ...]] = None,
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
        xy_scale: Optional[Tuple[float, ...]] = None,
        z_scale: Optional[Tuple[float, ...]] = None,
        num_levels: Optional[int] = None,
        memory_target: Optional[int] = None,
        start_T_src: Optional[int] = None,
        start_T_dest: Optional[int] = None,
        tbatch: Optional[int] = None,
        dtype: Optional[Union[str, np.dtype]] = None,
        auto_dask_cluster: bool = False,
    ) -> None:
        """
        Initialize an OME-Zarr converter with flexible scene selection,
        pyramid construction, and chunk-sizing.
        Parameters
        ----------
        source : str
            Path to the input image (any format supported by BioImage).
        destination : str
            Directory in which to write the ``.ome.zarr`` output(s).
        scenes : Optional[Union[int, List[int]]]
            Which scene(s) to export:
            - ``None`` → export all scenes
            - ``int``  → a single scene index
            - ``List[int]`` → those specific scene indices
        name : Optional[str]
            Base name for output files (defaults to the source stem). When exporting
            multiple scenes, each file name is suffixed with the scene’s name.
        scale : Optional[Tuple[Tuple[float, ...], ...]]
            Explicit per-level, per-axis relative sizes (one tuple per additional
            resolution level). Length of each tuple must match the native axis count.
            If provided, helper options below are ignored.
        chunk_shape : Optional[Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]]
            Chunk shape for Zarr arrays. Either a single shape applied to all levels
            (e.g., ``(1, 1, 16, 256, 256)``) or per-level shapes. Writer validates.
        shard_factor : Optional[Tuple[int, ...]]
            Optional shard factor per axis (Zarr v3 only). Writer validates.
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
        xy_scale : Optional[Tuple[float, ...]]
            Convenience: XY downsampling factors per additional level (e.g.,
            ``(0.5, 0.25)`` for 2 extra levels at 50% and 25% in X/Y). Applies only
            to present ``x``/``y`` axes. Ignored when ``scale`` is provided.
        z_scale : Optional[Tuple[float, ...]]
            Convenience: Z downsampling factors per level. Applies only if a ``z``
            axis is present. Ignored when ``scale`` is provided.
        num_levels : Optional[int]
            Convenience: build an XY half-pyramid with ``num_levels`` total levels
            (1 = level 0 only). Applies only to present ``x``/``y`` axes. Ignored
            when ``scale`` is provided.
        memory_target : Optional[int]
            If set (bytes), suggests a single chunk shape derived from level-0 shape
            and ``dtype`` via ``chunk_size_from_memory_target``. Writer may reuse or
            adjust per level.
        start_T_src : Optional[int]
            Source T index at which to begin reading from the BioImage. Default: use
            writer default.
        start_T_dest : Optional[int]
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
            conversion work begins. Default is False.
        """
        self.source = source
        self.destination = destination
        self.output_basename = name or Path(source).stem

        # spin up a local Dask cluster with 8 workers
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
        self._writer_scale = scale
        self._writer_chunk_shape = chunk_shape
        self._writer_shard_factor = shard_factor
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
        self._helper_xy_scale = xy_scale
        self._helper_z_scale = z_scale
        self._helper_num_levels = num_levels
        self._helper_memory_target_bytes = (
            None if memory_target is None else int(memory_target)
        )
        self._start_T_src = start_T_src
        self._start_T_dest = start_T_dest
        self._tbatch = None if tbatch is None else int(tbatch)

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
        if "c" not in axis_names:
            return None
        if self._writer_channels is not None:
            return self._writer_channels
        labels = self.bioimage.channel_names or [
            f"Channel:{i}" for i in range(channel_count)
        ]
        return [Channel(label=lab, color="#FFFFFF") for lab in labels[:channel_count]]

    def _build_scales_from_helpers(
        self, axis_names: List[str]
    ) -> Optional[Tuple[Tuple[float, ...], ...]]:
        # xy/z lists
        if self._helper_xy_scale is not None or self._helper_z_scale is not None:
            xs_in: Tuple[float, ...] = self._helper_xy_scale or ()
            zs_in: Tuple[float, ...] = self._helper_z_scale or ()
            nlevels = max(len(xs_in), len(zs_in))
            if nlevels == 0:
                return None

            # Local lists for padding levels
            xs: List[float] = list(xs_in) + [1.0] * (nlevels - len(xs_in))
            zs: List[float] = list(zs_in) + [1.0] * (nlevels - len(zs_in))

            per_level_xy: List[Tuple[float, ...]] = []
            for i in range(nlevels):
                vec: List[float] = []
                for ax in axis_names:
                    if ax in ("x", "y"):
                        vec.append(float(xs[i]))
                    elif ax == "z":
                        vec.append(float(zs[i]))
                    else:
                        vec.append(1.0)
                per_level_xy.append(tuple(vec))
            return tuple(per_level_xy)

        # simple XY half pyramid
        if self._helper_num_levels and self._helper_num_levels > 1:
            per_level_num: List[Tuple[float, ...]] = []
            for lvl in range(1, self._helper_num_levels):
                f = 0.5**lvl
                vec_tup: Tuple[float, ...] = tuple(
                    (f if ax in ("x", "y") else 1.0) for ax in axis_names
                )
                per_level_num.append(vec_tup)
            return tuple(per_level_num)

        return None

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

            # (3) Scale to writer
            writer_scale_param = (
                self._writer_scale
                if self._writer_scale is not None
                else self._build_scales_from_helpers(axis_names)
            )

            # (4) Chunking
            if self._writer_chunk_shape is not None:
                writer_chunk_shape_param: Optional[
                    ChunkShape
                ] = self._writer_chunk_shape
            elif self._helper_memory_target_bytes is not None:
                suggested = chunk_size_from_memory_target(
                    level0_shape,
                    self.output_dtype,
                    self._helper_memory_target_bytes,
                )
                writer_chunk_shape_param = tuple(int(x) for x in suggested)
            else:
                writer_chunk_shape_param = None

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

            # (6) Axis metadata (only for present axes)
            channel_models = self._resolve_channels(axis_names, ccount)
            pps = self._infer_physical_pixel_sizes(axis_names)

            writer_kwargs: Dict[str, Any] = {
                "store": str(out_path),
                "shape": level0_shape,
                "dtype": self.output_dtype,
                "image_name": (self._writer_image_name or base),
                "axes_names": (self._writer_axes_names or axis_names),
                "physical_pixel_size": pps,
                **{
                    k: v
                    for k, v in {
                        "scale": writer_scale_param,
                        "chunk_shape": writer_chunk_shape_param,
                        "shard_factor": self._writer_shard_factor,
                        "compressor": self._writer_compressor,
                        "zarr_format": self._writer_zarr_format,
                        "channels": channel_models,
                        "rdefs": self._writer_rdefs,
                        "creator_info": self._writer_creator_info,
                        "root_transform": self._writer_root_transform,
                        "axes_types": self._writer_axes_types,
                        "axes_units": self._writer_axes_units,
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

            # (8) Write
            has_t = "t" in axis_names
            T_total = int(getattr(r.dims, "T", 1)) if has_t else 1

            if has_t and T_total > 1:
                kwargs: Dict[str, Any] = {"data": data_all}
                if self._start_T_src is not None:
                    kwargs["start_T_src"] = self._start_T_src
                if self._start_T_dest is not None:
                    kwargs["start_T_dest"] = self._start_T_dest
                kwargs["total_T"] = (
                    self._tbatch if self._tbatch is not None else T_total
                )
                writer.write_timepoints(**kwargs)
            else:
                writer.write_full_volume(data_all)
