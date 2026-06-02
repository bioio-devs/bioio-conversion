from math import floor


def choose_zarr_layout(
    tczyx_shape: tuple[int, int, int, int, int],
    dtype_size: int,
    chunk_limit_bytes: int = 16 * 1024**2,
    shard_limit_bytes: int = 2 * 1024**3,
) -> tuple[
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
]:
    """
    Compute chunk and shard shapes for a TCZYX image.

    Parameters
    ----------
    tczyx_shape
        (T, C, Z, Y, X)

    dtype_size
        Number of bytes per voxel.

    chunk_limit_bytes
        Maximum uncompressed chunk size.

    shard_limit_bytes
        Maximum uncompressed shard size.

    Returns
    -------
    chunk_shape
        (T, C, Z, Y, X)

    shard_shape
        (T, C, Z, Y, X)

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

    T, C, Z, Y, X = tczyx_shape

    # ------------------------------------------------------------------
    # Determine chunk shape
    # ------------------------------------------------------------------

    bytes_per_slice = Y * X * dtype_size

    if bytes_per_slice > chunk_limit_bytes:
        raise ValueError(
            f"Single XY plane requires {bytes_per_slice:,} bytes "
            f"which exceeds chunk limit {chunk_limit_bytes:,} bytes."
        )

    z_chunk = chunk_limit_bytes // bytes_per_slice
    z_chunk = max(1, min(Z, z_chunk))

    chunk_shape = (
        1,
        1,
        z_chunk,
        Y,
        X,
    )

    chunk_bytes = z_chunk * bytes_per_slice

    # ------------------------------------------------------------------
    # Determine shard shape
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

    shard_shape = (
        shard_t,
        shard_c,
        shard_z_chunks * z_chunk,
        Y,
        X,
    )

    return chunk_shape, shard_shape


if __name__ == "__main__":
    dtype_size = 2
    chunk_limit = 16 * 1024**2  # 16 MiB
    shard_limit = 4 * 1024**3  # 4 GiB

    shape = (100, 4, 100, 1500, 2500)
    T, C, Z, Y, X = shape

    atlas_size = 2048

    total_chunks = 0
    total_shards = 0

    level = 0
    while True:
        current_shape = (T, C, Z, Y, X)
        chunk_shape, shard_shape = choose_zarr_layout(
            tczyx_shape=current_shape,
            dtype_size=dtype_size,
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
            f"  atlas tiles:  {tiles_x} x {tiles_y} = {tiles_x * tiles_y} "
            f"(need {Z}, fits={fits_atlas})"
        )

        # Stop once Z slices of size (Y, X) tile into a 2048x2048 atlas.
        if fits_atlas:
            break

        # Downsample XY by half until one of X or Y is less than Z,
        # then downsample Z by half.
        if min(X, Y) >= Z:
            X = max(1, X // 2)
            Y = max(1, Y // 2)
        else:
            Z = max(1, Z // 2)

        level += 1

    print()
    print(f"total chunks across all levels: {total_chunks:,}")
    print(f"total shards across all levels: {total_shards:,}")
