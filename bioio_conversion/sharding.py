from typing import Any, List, Tuple, Union

import numpy as np
from bioio_base.dimensions import DimensionNames
from bioio_ome_zarr.writers.utils import multiscale_chunk_size_from_memory_target

ATLAS_SIZE = 2048

# Default uncompressed size budgets for the Zarr v3 layout. Single source of
# truth so callers (e.g. the converter) don't re-hardcode the literals.
DEFAULT_CHUNK_LIMIT_BYTES = 16 * 1024**2
DEFAULT_SHARD_LIMIT_BYTES = 4 * 1024**3


def _round_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _choose_zarr_layout(
    shape: Tuple[int, ...],
    dtype: Union[str, "np.dtype[Any]"],
    dims: str,
    chunk_limit_bytes: int = DEFAULT_CHUNK_LIMIT_BYTES,
    shard_limit_bytes: int = DEFAULT_SHARD_LIMIT_BYTES,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute chunk and shard shapes for a Zarr v3 OME-Zarr array.

    Intended for level-0 of a multi-resolution pyramid.  For lower levels use
    ``_proportional_shard`` to derive a shard that keeps the number of shards
    per axis constant, which is required for safe parallel region writes.

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

    chunk_raw = tuple(
        multiscale_chunk_size_from_memory_target([shape], dtype, chunk_limit_bytes)[0]
    )
    # Force single-T, single-C chunks; keep Z/Y/X from the memory-target helper.
    chunk_map = {
        **dict(zip(dims, chunk_raw)),
        DimensionNames.Time: 1,
        DimensionNames.Channel: 1,
    }
    chunk_shape = tuple(chunk_map[d] for d in dims)
    z_chunk = chunk_map.get(DimensionNames.SpatialZ, 1)
    chunk_y = chunk_map[DimensionNames.SpatialY]
    chunk_x = chunk_map[DimensionNames.SpatialX]

    # Pack chunks along Z→Y→X→C→T up to the shard budget (C and T are one
    # chunk wide, so their unit is 1).
    chunk_bytes = int(np.prod(chunk_shape)) * np.dtype(dtype).itemsize
    max_chunks_per_shard = max(1, shard_limit_bytes // chunk_bytes)

    used = 1
    shard_sizes = {}
    for d, count, unit in (
        (DimensionNames.SpatialZ, (Z + z_chunk - 1) // z_chunk, z_chunk),
        (DimensionNames.SpatialY, (Y + chunk_y - 1) // chunk_y, chunk_y),
        (DimensionNames.SpatialX, (X + chunk_x - 1) // chunk_x, chunk_x),
        (DimensionNames.Channel, C, 1),
        (DimensionNames.Time, T, 1),
    ):
        take = max(1, min(count, max_chunks_per_shard // used))
        shard_sizes[d] = take * unit
        used *= take

    shard_shape = tuple(shard_sizes[d] for d in dims)

    return chunk_shape, shard_shape


def _proportional_shard(
    shape: Tuple[int, ...],
    chunk_shape: Tuple[int, ...],
    reference_shape: Tuple[int, ...],
    reference_shard: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Derive a shard shape for a lower pyramid level proportional to level 0.

    Each axis is scaled as ``reference_shard[ax] * shape[ax] / reference_shape[ax]``
    then rounded up to the nearest chunk multiple.  This keeps the number of
    shards per axis constant across all pyramid levels, which is required for
    safe parallel region writes via ``write_region``.

    The result is additionally capped at the minimum chunk-multiple that covers
    the actual axis extent (``ceil(shape[ax] / chunk[ax]) * chunk[ax]``).
    Without this cap, an overrun at level 0 (where the shard is a chunk-multiple
    that exceeds the real dimension) propagates and compounds at lower levels,
    producing shards with phantom overflow chunks that can never be filled — a
    region write would therefore always produce an incomplete shard.

    Parameters
    ----------
    shape
        Level shape to compute a shard for.
    chunk_shape
        Chunk shape at this level (from ``_choose_zarr_layout``).
    reference_shape
        Level-0 shape (proportionality baseline).
    reference_shard
        Level-0 shard shape (proportionality baseline).

    Returns
    -------
    shard_shape
        In the same axis order and length as ``shape``.
    """
    shard_shape = []
    for ax in range(len(shape)):
        # Scale the level-0 shard proportionally to this level's extent, but
        # never below a single chunk, then round up to a whole chunk multiple.
        proportional = round(reference_shard[ax] * shape[ax] / reference_shape[ax])
        scaled = _round_to_multiple(max(chunk_shape[ax], proportional), chunk_shape[ax])

        # Cap at the smallest chunk multiple that covers the actual extent so we
        # don't carry phantom overflow chunks that could never be filled.
        coverage_cap = _round_to_multiple(shape[ax], chunk_shape[ax])

        shard_shape.append(min(scaled, coverage_cap))
    return tuple(shard_shape)


def choose_pyramid_layout(
    level_shapes: List[Tuple[int, ...]],
    dtype: Union[str, "np.dtype[Any]"],
    dims: str,
    chunk_limit_bytes: int = DEFAULT_CHUNK_LIMIT_BYTES,
    shard_limit_bytes: int = DEFAULT_SHARD_LIMIT_BYTES,
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """
    Compute chunk and shard shapes for every level of a multi-resolution pyramid.

    Level 0 uses the budget shard algorithm (``_choose_zarr_layout``).  All
    subsequent levels use ``_proportional_shard`` to keep the number of shards
    per axis constant across the pyramid (see that function for why).

    Parameters
    ----------
    level_shapes
        Per-level shapes, level 0 first.
    dtype
        Array dtype (any form accepted by ``np.dtype``).
    dims
        Dimension labels matching the axis order of the shapes.
    chunk_limit_bytes
        Maximum uncompressed chunk size. Default: 16 MiB.
    shard_limit_bytes
        Maximum uncompressed shard size for level 0. Default: 4 GiB.

    Returns
    -------
    all_chunks, all_shards
        Per-level chunk and shard shapes, level 0 first.
    """
    chunk0, shard0 = _choose_zarr_layout(
        level_shapes[0], dtype, dims, chunk_limit_bytes, shard_limit_bytes
    )
    all_chunks: List[Tuple[int, ...]] = [chunk0]
    all_shards: List[Tuple[int, ...]] = [shard0]

    shape0 = level_shapes[0]
    for lvl_shape in level_shapes[1:]:
        lvl_chunk_raw, _ = _choose_zarr_layout(
            lvl_shape, dtype, dims, chunk_limit_bytes
        )
        # Cap each chunk axis to the proportional shard target.  Without this,
        # chunk sizes grow as levels get smaller (the budget fills more of the
        # axis), eventually exceeding the proportional target and forcing
        # _proportional_shard to pick a larger shard — breaking the constant
        # n_shards invariant.
        prop_target = tuple(
            max(1, round(shard0[ax] * lvl_shape[ax] / shape0[ax]))
            for ax in range(len(lvl_shape))
        )
        lvl_chunk = tuple(
            min(lvl_chunk_raw[ax], prop_target[ax]) for ax in range(len(lvl_shape))
        )
        all_chunks.append(lvl_chunk)
        all_shards.append(_proportional_shard(lvl_shape, lvl_chunk, shape0, shard0))

    return all_chunks, all_shards


def build_pyramid_shapes(
    base_shape: Tuple[int, ...],
    dims: str,
    atlas_size: int = ATLAS_SIZE,
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
        Default: ``ATLAS_SIZE`` (2048).

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
