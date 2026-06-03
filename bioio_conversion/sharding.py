from typing import Any, List, Tuple, Union

import numpy as np
from bioio_base.dimensions import DimensionNames
from bioio_ome_zarr.writers.utils import multiscale_chunk_size_from_memory_target

_ATLAS_SIZE = 2048


def _choose_zarr_layout(
    shape: Tuple[int, ...],
    dtype: Union[str, "np.dtype[Any]"],
    dims: str,
    chunk_limit_bytes: int = 16 * 1024**2,
    shard_limit_bytes: int = 4 * 1024**3,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute chunk and shard shapes for a Zarr v3 OME-Zarr array.

    Parameters
    ----------
    shape
        Image shape matching ``dims``.
    dtype
        Array dtype (any form accepted by ``np.dtype``).
    dims
        Dimension labels for ``shape`` in reader-native order.  Axis order
        is preserved in the returned shapes.
    chunk_limit_bytes
        Maximum uncompressed chunk size. Default: 16 MiB.
    shard_limit_bytes
        Maximum uncompressed shard size. Default: 4 GiB.

    Returns
    -------
    chunk_shape, shard_shape
        Both in the same axis order and length as ``shape``.
    """
    dim_sizes = dict(zip(dims, shape))
    T = dim_sizes.get(DimensionNames.Time, 1)
    C = dim_sizes.get(DimensionNames.Channel, 1)
    Z = dim_sizes.get(DimensionNames.SpatialZ, 1)
    Y = dim_sizes[DimensionNames.SpatialY]
    X = dim_sizes[DimensionNames.SpatialX]

    # Chunk computation: delegate to multiscale_chunk_size_from_memory_target,
    chunk_raw = tuple(
        multiscale_chunk_size_from_memory_target([shape], dtype, chunk_limit_bytes)[0]
    )
    chunk_raw_map = dict(zip(dims, chunk_raw))
    chunk_y = chunk_raw_map[DimensionNames.SpatialY]
    chunk_x = chunk_raw_map[DimensionNames.SpatialX]
    z_chunk = chunk_raw_map.get(DimensionNames.SpatialZ, 1)

    chunk_sizes = {
        DimensionNames.Time: 1,
        DimensionNames.Channel: 1,
        DimensionNames.SpatialZ: z_chunk,
        DimensionNames.SpatialY: chunk_y,
        DimensionNames.SpatialX: chunk_x,
    }
    chunk_shape = tuple(chunk_sizes[d] for d in dims)

    # Shard computation: pack chunks along Z→Y→X→C→T up to the shard budget.
    chunk_bytes = int(np.prod(chunk_shape)) * np.dtype(dtype).itemsize
    max_chunks_per_shard = max(1, shard_limit_bytes // chunk_bytes)

    z_chunk_count = (Z + z_chunk - 1) // z_chunk
    shard_z_chunks = min(z_chunk_count, max_chunks_per_shard)
    chunks_used = shard_z_chunks

    y_chunk_count = (Y + chunk_y - 1) // chunk_y
    shard_y_chunks = min(y_chunk_count, max_chunks_per_shard // chunks_used)
    chunks_used *= shard_y_chunks

    x_chunk_count = (X + chunk_x - 1) // chunk_x
    shard_x_chunks = min(x_chunk_count, max_chunks_per_shard // chunks_used)
    chunks_used *= shard_x_chunks

    shard_c = max(1, min(C, max_chunks_per_shard // chunks_used))
    chunks_used *= shard_c

    shard_t = max(1, min(T, max_chunks_per_shard // chunks_used))

    shard_sizes = {
        DimensionNames.Time: shard_t,
        DimensionNames.Channel: shard_c,
        DimensionNames.SpatialZ: shard_z_chunks * z_chunk,
        DimensionNames.SpatialY: shard_y_chunks * chunk_y,
        DimensionNames.SpatialX: shard_x_chunks * chunk_x,
    }
    shard_shape = tuple(shard_sizes[d] for d in dims)

    return chunk_shape, shard_shape


def _build_pyramid_shapes(
    base_shape: Tuple[int, ...],
    dims: str,
    atlas_size: int = _ATLAS_SIZE,
) -> List[Tuple[int, ...]]:
    """
    Generate multi-resolution pyramid level shapes with atlas-fit termination.

    Downsamples only Z, Y, and X — T and C are never changed.  Halves X and Y
    together while ``min(X, Y) >= Z``; once that condition no longer holds,
    halves Z instead.  Stops as soon as all Z slices of the current level can
    be tiled into an ``atlas_size × atlas_size`` canvas:
    ``(atlas_size // X) * (atlas_size // Y) >= Z``.

    Parameters
    ----------
    base_shape
        Level-0 image shape matching ``dims``.
    dims
        Dimension labels for ``base_shape`` in reader-native order.
    atlas_size
        Edge length (in pixels) of the square viewer atlas canvas.
        Default: ``_ATLAS_SIZE`` (2048).

    Returns
    -------
    List[Tuple[int, ...]]
        Per-level shapes, level 0 first, each in the same axis order as
        ``dims``.  The list always contains at least one entry.
    """
    dim_map = dict(zip(dims, base_shape))
    T = dim_map.get(DimensionNames.Time, 1)
    C = dim_map.get(DimensionNames.Channel, 1)
    Z = dim_map.get(DimensionNames.SpatialZ, 1)
    Y = dim_map[DimensionNames.SpatialY]
    X = dim_map[DimensionNames.SpatialX]

    levels: List[Tuple[int, ...]] = []

    while True:
        current = {
            DimensionNames.Time: T,
            DimensionNames.Channel: C,
            DimensionNames.SpatialZ: Z,
            DimensionNames.SpatialY: Y,
            DimensionNames.SpatialX: X,
        }
        levels.append(tuple(current[d] for d in dims))

        tiles_x = atlas_size // X if X <= atlas_size else 0
        tiles_y = atlas_size // Y if Y <= atlas_size else 0
        if tiles_x * tiles_y >= Z:
            break

        if min(X, Y) >= Z:
            X = max(1, X // 2)
            Y = max(1, Y // 2)
        else:
            Z = max(1, Z // 2)

    return levels
