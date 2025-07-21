from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from bioio import BioImage
from bioio_ome_zarr.writers import OmeZarrWriterV2
from bioio_ome_zarr.writers.utils import DimTuple, chunk_size_from_memory_target


class OmeZarrConverter:
    def __init__(
        self,
        *,
        source: str,
        destination: str,
        scene: int = 0,
        name: Optional[str] = None,
        overwrite: bool = False,
        tbatch: int = 1,
        level_scales: List[DimTuple] = [(1.0, 1.0, 1.0, 1.0, 1.0)],
        dtype: Optional[Union[str, np.dtype]] = None,
        channel_names: Optional[List[str]] = None,
    ):
        self.source = source
        self.destination = destination
        self.scene = scene
        self.name = name or Path(source).stem
        self.overwrite = overwrite
        self.tbatch = tbatch
        self.level_scales = level_scales

        # Pre-load BioImage to capture dims and defaults
        bio = BioImage(self.source)
        dims = bio.dims
        self.source_shape = (dims.T, dims.C, dims.Z, dims.Y, dims.X)
        self.dtype = np.dtype(dtype) if dtype is not None else bio.dtype
        self.channels = list(range(dims.C))
        self.channel_names = (
            channel_names
            if channel_names is not None
            else (bio.channel_names or [f"Channel:{i}" for i in self.channels])
        )

    @staticmethod
    def get_level_shapes(
        source_shape: DimTuple,
        level_scales: List[DimTuple],
        channels: Optional[List[int]] = None,
    ) -> List[DimTuple]:
        t0, c0, z0, y0, x0 = source_shape
        c = len(channels) if channels is not None else c0

        shapes: List[DimTuple] = []
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
        memory_target: int = 16 * 1024 * 1024,
    ) -> List[DimTuple]:
        raw = [
            chunk_size_from_memory_target(shape, np.dtype("uint8"), memory_target)
            for shape in level_shapes
        ]
        # clamp each dimension to at least 1
        return [tuple(max(1, d) for d in dims) for dims in raw]

    def convert(self) -> None:
        # 1) Load via BioImage and set the configured scene
        bioimg = BioImage(self.source)
        bioimg.set_scene(self.scene)

        # 2) Refresh dims and compute dtype + channel_names from parameters
        dims = bioimg.dims
        self.source_shape = (dims.T, dims.C, dims.Z, dims.Y, dims.X)
        dtype = self.dtype
        channel_names = self.channel_names

        # 3) Gather physical spacing from BioImage
        scale = bioimg.scale
        t_s = scale.T or 1.0
        z_s = scale.Z or 1.0
        y_s = scale.Y or 1.0
        x_s = scale.X or 1.0
        physical_dims = {"t": t_s, "z": z_s, "y": y_s, "x": x_s, "c": 1.0}
        physical_units = {
            "t": "second",
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
        }

        # 4) Prepare/clean output path
        full_path = Path(self.destination) / f"{self.name}.ome.zarr"
        if full_path.exists():
            if self.overwrite:
                import shutil

                shutil.rmtree(full_path)
            else:
                raise FileExistsError(
                    f"{full_path} exists; use overwrite=True to overwrite."
                )

        # 5) Compute shapes & chunks
        level_shapes = self.get_level_shapes(
            self.source_shape, self.level_scales, self.channels
        )
        chunk_dims = self.get_zarr_chunk_dims(level_shapes)

        # 6) Initialize Zarr-V2 writer
        writer = OmeZarrWriterV2()
        writer.init_store(
            output_path=str(full_path),
            shapes=level_shapes,
            chunk_sizes=chunk_dims,
            dtype=dtype,
        )

        # 7) Write pixel data in time-batches
        writer.write_t_batches(
            bioimg.reader,
            channels=self.channels,
            tbatch=self.tbatch,
            debug=False,
        )

        # 8) Generate & write metadata
        meta = writer.generate_metadata(
            image_name=self.name,
            channel_names=channel_names,
            physical_dims=physical_dims,
            physical_units=physical_units,
            channel_colors=[0xFFFFFF] * dims.C,
        )
        writer.write_metadata(meta)
