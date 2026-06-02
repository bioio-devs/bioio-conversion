from math import floor

_ALLOWED_DIMS = ("T", "C", "Z", "Y", "X")


def _validate_dims(dims: str, shape: tuple[int, ...]) -> None:
    if len(dims) != len(shape):
        raise ValueError(
            f"dims {dims!r} length does not match shape length {len(shape)}."
        )
    if not (2 <= len(dims) <= 5):
        raise ValueError(f"dims must have 2 to 5 entries, got {dims!r}.")
    if dims[-2:] != "YX":
        raise ValueError(f"Last two dims must be 'YX', got {dims!r}.")
    leading = dims[:-2]
    if any(d not in {"T", "C", "Z"} for d in leading):
        raise ValueError(f"Leading dims must be from {{T, C, Z}}, got {dims!r}.")
    if len(set(leading)) != len(leading):
        raise ValueError(f"Duplicate dims in {dims!r}.")


def _expand_to_tczyx(
    dims: str, shape: tuple[int, ...]
) -> tuple[int, int, int, int, int]:
    """Expand an arbitrary (2D-5D) shape with dims label to full TCZYX."""
    mapping = dict(zip(dims, shape))
    return tuple(mapping.get(d, 1) for d in _ALLOWED_DIMS)  # type: ignore[return-value]


def _contract_from_tczyx(
    dims: str, tczyx_shape: tuple[int, int, int, int, int]
) -> tuple[int, ...]:
    """Project a TCZYX-ordered shape down to the user's dims order."""
    full = dict(zip(_ALLOWED_DIMS, tczyx_shape))
    return tuple(full[d] for d in dims)


def choose_zarr_layout(
    shape: tuple[int, ...],
    dtype_size: int,
    dims: str = "TCZYX",
    chunk_limit_bytes: int = 16 * 1024**2,
    shard_limit_bytes: int = 2 * 1024**3,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Compute chunk and shard shapes for a 2D-5D image.

    Parameters
    ----------
    shape
        Image shape, length 2 to 5, matching ``dims``.

    dtype_size
        Number of bytes per voxel.

    dims
        Dimension labels for ``shape``. Must end in ``"YX"`` and the
        leading dims must be a subset of ``{"T", "C", "Z"}`` in any
        order. Examples: ``"YX"``, ``"ZYX"``, ``"CYX"``, ``"CZYX"``,
        ``"TCZYX"``.

    chunk_limit_bytes
        Maximum uncompressed chunk size.

    shard_limit_bytes
        Maximum uncompressed shard size.

    Returns
    -------
    chunk_shape
        Same length and order as ``shape``.

    shard_shape
        Same length and order as ``shape``.

    Notes
    -----
    Assumptions:

    - Viewer reads whole Z slices.
    - Channels and timepoints are accessed independently.
    - Chunking only occurs along Z.
    - Chunks contain complete XY planes.
    - Shards are integer multiples of chunks.
    - Shards are grown preferentially along Z, then C, then T.
    """

    _validate_dims(dims, shape)

    T, C, Z, Y, X = _expand_to_tczyx(dims, shape)

    # ------------------------------------------------------------------
    # Determine chunk shape.
    # Chunk shape is designed around a viewer's access pattern of
    # whole Z slices at a single channel at a single timepoint.
    # ------------------------------------------------------------------

    bytes_per_slice = Y * X * dtype_size

    if bytes_per_slice > chunk_limit_bytes:
        raise ValueError(
            f"Single XY plane requires {bytes_per_slice:,} bytes "
            f"which exceeds chunk limit {chunk_limit_bytes:,} bytes."
        )

    z_chunk = chunk_limit_bytes // bytes_per_slice
    z_chunk = max(1, min(Z, z_chunk))

    chunk_tczyx = (1, 1, z_chunk, Y, X)

    chunk_bytes = z_chunk * bytes_per_slice

    # ------------------------------------------------------------------
    # Determine shard shape
    # Shard shape is designed around simply packing as many chunks as
    # possible into each shard, up to some size limit.
    # ------------------------------------------------------------------

    max_chunks_per_shard = max(
        1,
        shard_limit_bytes // chunk_bytes,
    )

    # Number of chunk positions along Z
    z_chunk_count = (Z + z_chunk - 1) // z_chunk

    # Start with one chunk
    shard_t = 1
    shard_c = 1
    shard_z_chunks = 1

    chunks_used = 1

    # --------------------------------------------------------------
    # Grow along Z first
    # --------------------------------------------------------------

    max_z_chunks_in_shard = min(
        z_chunk_count,
        max_chunks_per_shard,
    )

    shard_z_chunks = max_z_chunks_in_shard
    chunks_used = shard_z_chunks

    # --------------------------------------------------------------
    # Grow along C second
    # --------------------------------------------------------------

    max_c = min(
        C,
        max_chunks_per_shard // chunks_used,
    )

    shard_c = max(1, max_c)
    chunks_used *= shard_c

    # --------------------------------------------------------------
    # Grow along T third
    # --------------------------------------------------------------

    max_t = min(
        T,
        max_chunks_per_shard // chunks_used,
    )

    shard_t = max(1, max_t)
    chunks_used *= shard_t

    shard_tczyx = (
        shard_t,
        shard_c,
        shard_z_chunks * z_chunk,
        Y,
        X,
    )

    chunk_shape = _contract_from_tczyx(dims, chunk_tczyx)
    shard_shape = _contract_from_tczyx(dims, shard_tczyx)

    return chunk_shape, shard_shape


if __name__ == "__main__":
    dtype_size = 2
    chunk_limit = 16 * 1024**2  # 16 MiB
    shard_limit = 4 * 1024**3  # 4 GiB

    atlas_size = 2048

    examples = [
        ("YX", (1500, 2500)),
        ("ZYX", (1000, 1500, 2500)),
        ("CYX", (4, 1500, 2500)),
        ("CZYX", (4, 1000, 1500, 2500)),
        ("TCZYX", (100, 4, 1000, 1500, 2500)),
    ]

    for dims, shape in examples:
        print(f"=== dims={dims} shape={shape} ===")

        sizes = dict(zip(dims, shape))
        T = sizes.get("T", 1)
        C = sizes.get("C", 1)
        Z = sizes.get("Z", 1)
        Y = sizes["Y"]
        X = sizes["X"]

        total_chunks = 0
        total_shards = 0

        level = 0
        while True:
            current_full = (T, C, Z, Y, X)
            current_shape = _contract_from_tczyx(dims, current_full)

            chunk_shape, shard_shape = choose_zarr_layout(
                shape=current_shape,
                dtype_size=dtype_size,
                dims=dims,
                chunk_limit_bytes=chunk_limit,
                shard_limit_bytes=shard_limit,
            )

            chunk_bytes = dtype_size
            for d in chunk_shape:
                chunk_bytes *= d

            shard_bytes = dtype_size
            for d in shard_shape:
                shard_bytes *= d

            n_chunks = 1
            for dim, c in zip(current_shape, chunk_shape):
                n_chunks *= (dim + c - 1) // c

            n_shards = 1
            for dim, s in zip(current_shape, shard_shape):
                n_shards *= (dim + s - 1) // s

            total_chunks += n_chunks
            total_shards += n_shards

            tiles_x = atlas_size // X if X <= atlas_size else 0
            tiles_y = atlas_size // Y if Y <= atlas_size else 0
            fits_atlas = tiles_x * tiles_y >= Z

            print(f"level {level}:")
            print(f"  shape:        {current_shape}")
            print(f"  chunk_shape:  {chunk_shape}  ({chunk_bytes:,} bytes)")
            print(f"  shard_shape:  {shard_shape}  ({shard_bytes:,} bytes)")
            print(f"  n_chunks:     {n_chunks:,}")
            print(f"  n_shards:     {n_shards:,}")
            print(
                f"  atlas tiles:  {tiles_x} x {tiles_y} = "
                f"{tiles_x * tiles_y} (need {Z}, fits={fits_atlas})"
            )

            # Stop once Z slices of size (Y, X) tile into the atlas.
            if fits_atlas:
                break

            # Downsample only spatial dims (Z, Y, X). Never T or C.
            # Halve XY until one of X or Y is below Z, then halve Z.
            if min(X, Y) >= Z:
                X = max(1, X // 2)
                Y = max(1, Y // 2)
            else:
                Z = max(1, Z // 2)

            level += 1

        print()
        print(f"total chunks across all levels: {total_chunks:,}")
        print(f"total shards across all levels: {total_shards:,}")
        print()
