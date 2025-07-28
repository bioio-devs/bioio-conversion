import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from bioio import BioImage
from bioio_ome_zarr.writers import OmeZarrWriterV2
from bioio_ome_zarr.writers.utils import DimTuple, chunk_size_from_memory_target


class OmeZarrConverter:
    """
    OmeZarrConverter handles conversion of any BioImage‐supported format
    (TIFF, CZI, etc.) into OME-Zarr v2 stores. Supports exporting one,
    many, or all scenes from a multi-scene file, with configurable
    multi-resolution pyramids (full TCZYX scales or per-axis XY/Z
    factors) and chunk-size memory targeting.
    """

    def __init__(
        self,
        *,
        source: str,
        destination: str,
        scenes: Union[int, List[int]] = 0,
        name: Optional[str] = None,
        overwrite: bool = False,
        tbatch: int = 1,
        level_scales: Optional[List[DimTuple]] = None,
        xy_scale: Optional[Tuple[float, ...]] = None,
        z_scale: Optional[Tuple[float, ...]] = None,
        memory_target: int = 16 * 1024 * 1024,
        dtype: Optional[Union[str, np.dtype]] = None,
        channel_names: Optional[List[str]] = None,
    ):
        """
        Initialize an OME-Zarr converter with flexible scene selection,
        pyramid construction, and chunk-sizing.

        Parameters
        ----------
        source : str
            Path to the input image (any format supported by BioImage).
        destination : str
            Directory in which to write `.ome.zarr` outputs.
        scenes : Union[int, List[int]]
            Which scene(s) to export:
              - non-negative int → single scene index
              - list of ints     → those specific scene indices
              - -1                → all available scenes
        name : Optional[str]
            Base name for output files (defaults to source stem). When
            exporting multiple scenes, each file is suffixed with the
            scene’s name.
        overwrite : bool
            If True, remove any existing Zarr store at the target path.
        tbatch : int
            Number of timepoints per batch when streaming writes.
        level_scales : Optional[List[DimTuple]]
            Explicit list of (t, c, z, y, x) scale tuples—one per
            resolution level. Mutually exclusive with `xy_scale`/`z_scale`.
        xy_scale : Optional[Tuple[float, ...]]
            Downsampling factors for X and Y, relative to the original.
            E.g. `(0.5, 0.25)` → levels at 100%, 50%, 25% in X/Y.
        z_scale : Optional[Tuple[float, ...]]
            Downsampling factors for Z, relative to the original. Must
            match `xy_scale` length if provided. Can be used alone or
            together with `xy_scale`.
        memory_target : int
            Approximate maximum bytes per Zarr chunk. Passed to
            `chunk_size_from_memory_target` (default 16 MB).
        dtype : Optional[Union[str, np.dtype]]
            Override output data type (e.g. `"uint16"`); defaults to
            the reader’s dtype.
        channel_names : Optional[List[str]]
            Override channel labels in the metadata; defaults to the
            reader’s `channel_names` or generic IDs.

        Raises
        ------
        ValueError
            If `level_scales` is used together with `xy_scale`/`z_scale`,
            or if `xy_scale` and `z_scale` lengths mismatch.
        IndexError
            If any requested scene index is out of range.
        """
        self.source = source
        self.destination = destination
        self.raw_scenes = scenes
        self.name = name or Path(source).stem
        self.overwrite = overwrite
        self.tbatch = tbatch
        self.memory_target = memory_target

        bio_probe = BioImage(self.source)
        total = len(bio_probe.scenes)

        if isinstance(scenes, int):
            if scenes < 0:
                self.scenes = list(range(total))
            else:
                if scenes >= total:
                    raise IndexError(
                        f"Scene index {scenes} ≥ number of scenes ({total})"
                    )
                self.scenes = [scenes]
        else:
            invalid = [s for s in scenes if s < 0 or s >= total]
            if invalid:
                raise IndexError(f"Scene indices out of range: {invalid}")
            self.scenes = scenes.copy()

        # dtype & channel names come from scene 0 by default
        bio0 = BioImage(self.source)
        bio0.set_scene(self.scenes[0])
        self.dtype = np.dtype(dtype) if dtype is not None else bio0.dtype
        dims0 = bio0.dims
        self.channels = list(range(dims0.C))
        self.channel_names = (
            channel_names
            if channel_names is not None
            else (bio0.channel_names or [f"Channel:{i}" for i in self.channels])
        )

        # Build level_scales
        if level_scales is not None:
            if xy_scale or z_scale:
                raise ValueError("Cannot mix `level_scales` with `xy_scale`/`z_scale`.")
            self.level_scales = level_scales
        else:
            # fill per-axis tuples
            if xy_scale and not z_scale:
                z_scale = tuple(1.0 for _ in xy_scale)
            if z_scale and not xy_scale:
                xy_scale = tuple(1.0 for _ in z_scale)
            if xy_scale and z_scale and len(xy_scale) != len(z_scale):
                raise ValueError("`xy_scale` and `z_scale` must match length.")
            scales: List[DimTuple] = [(1, 1, 1, 1, 1)]
            if xy_scale and z_scale:
                for xy, z in zip(xy_scale, z_scale):
                    scales.append((1.0, 1.0, float(z), float(xy), float(xy)))
            self.level_scales = scales

    @staticmethod
    def get_level_shapes(
        source_shape: DimTuple,
        level_scales: List[DimTuple],
        channels: Optional[List[int]] = None,
    ) -> List[DimTuple]:
        t0, c0, z0, y0, x0 = source_shape
        c = len(channels) if channels is not None else c0
        shapes = []
        for st, _, sz, sy, sx in level_scales:
            shapes.append(
                (
                    int(round(t0 * st)),
                    c,
                    int(round(z0 * sz)),
                    int(round(y0 * sy)),
                    int(round(x0 * sx)),
                )
            )
        return shapes

    @staticmethod
    def get_zarr_chunk_dims(
        level_shapes: List[DimTuple],
        memory_target: int,
    ) -> List[DimTuple]:
        raw = [
            chunk_size_from_memory_target(shape, np.dtype("uint8"), memory_target)
            for shape in level_shapes
        ]
        return [tuple(max(1, d) for d in dims) for dims in raw]

    def convert(self) -> None:
        """
        Loop over each scene in self.scenes, writing
        an independent .ome.zarr file named:
            {self.name}_{scene_name}.ome.zarr
        """

        # This is really only for windows
        if len(self.scenes) > 1:
            invalid = []
            for i in self.scenes:
                name = BioImage(self.source).scenes[i]
                if re.search(r'[<>:"/\\|?*]', name):
                    invalid.append(name)
            if invalid:
                warnings.warn(
                    f"Scene names {invalid} contain invalid chars and will be "
                    "sanitized for the output file name.",
                    UserWarning,
                )

        for idx in self.scenes:  # This could be parallelized
            bio = BioImage(self.source)
            bio.set_scene(idx)
            dims = bio.dims
            shape5 = (dims.T, dims.C, dims.Z, dims.Y, dims.X)

            # physical scales (fallback to 1.0)
            scale = bio.scale
            phys = {
                "t": scale.T or 1.0,
                "z": scale.Z or 1.0,
                "y": scale.Y or 1.0,
                "x": scale.X or 1.0,
                "c": 1.0,
            }
            units = {
                "t": "second",
                "z": "micrometer",
                "y": "micrometer",
                "x": "micrometer",
            }

            # determine output name: append scene name if >1
            scene_name = bio.scenes[idx]

            # remove invalid chars
            out_name = re.sub(
                r'[<>:"/\\|?*]',
                "_",
                (self.name if len(self.scenes) == 1 else f"{self.name}_{scene_name}"),
            )
            full_path = Path(self.destination) / f"{out_name}.ome.zarr"

            # clean/overwrite
            if full_path.exists():
                if self.overwrite:
                    import shutil

                    shutil.rmtree(full_path)
                else:
                    raise FileExistsError(f"{full_path} exists; set overwrite=True.")

            # compute shapes & chunks
            lvl_shapes = self.get_level_shapes(shape5, self.level_scales, self.channels)
            chunk_dims = self.get_zarr_chunk_dims(lvl_shapes, self.memory_target)

            # write
            writer = OmeZarrWriterV2()
            writer.init_store(
                output_path=str(full_path),
                shapes=lvl_shapes,
                chunk_sizes=chunk_dims,
                dtype=self.dtype,
            )
            writer.write_t_batches(
                bio.reader,
                channels=self.channels,
                tbatch=self.tbatch,
                debug=False,
            )

            # metadata
            meta = writer.generate_metadata(
                image_name=out_name,
                channel_names=self.channel_names,
                physical_dims=phys,
                physical_units=units,
                channel_colors=[0xFFFFFF] * dims.C,
            )
            writer.write_metadata(meta)
