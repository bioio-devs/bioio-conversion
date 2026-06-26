# bioio-conversion

[![Build Status](https://github.com/bioio-devs/bioio-conversion/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-conversion/actions)
[![Documentation](https://github.com/bioio-devs/bioio-conversion/actions/workflows/docs.yml/badge.svg)](https://bioio-devs.github.io/bioio-conversion/overview.html)
[![PyPI version](https://badge.fury.io/py/bioio-conversion.svg)](https://badge.fury.io/py/bioio-conversion)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO conversion tool for going between image formats.

---

## Documentation

See the full documentation on our GitHub Pages site:

[https://bioio-devs.github.io/bioio-conversion](https://bioio-devs.github.io/bioio-conversion/overview.html)

---

## Installation

Install from PyPI along with core BioIO and plugins:

```bash
pip install bioio-conversion
```

---

## Python Package Usage

### Available Converters

* **OmeZarrConverter**

  * **Purpose**: Convert any BioImage-supported input (TIFF, CZI, ND2, etc.) into an OME-Zarr store.
  * **Features**:

    * Multi-scene export (`scenes=0`, list, or `None` = all)
    * Flexible multiscale pyramid options (`level_shapes`, `num_levels`, `downsample_z`)
    * Chunk-size tuning (`chunk_shape`, `memory_target`, `shard_shape`)
    * Metadata options (`channels`, `axes_names`, `axes_units`, `axes_types`, `physical_pixel_size`)
    * Output format (`zarr_format` = 2 or 3)
    * Optional auto Dask cluster
  * **Import path**:

    ```python
    from bioio_conversion.converters import OmeZarrConverter
    ```

* **BatchConverter**

  * **Purpose**: Orchestrate batch conversions of many files (CSV, directory crawl, or explicit list).
  * **Features**:

    * Factory methods: `from_csv()`, `from_directory()`, `from_list()`
    * Shared `default_opts` for per-job overrides
    * Dispatch jobs via `.run_jobs()`
  * **Import path**:

    ```python
    from bioio_conversion.converters import BatchConverter
    ```

---

### Example: OmeZarrConverter

#### Minimal usage

```python
from bioio_conversion.converters import OmeZarrConverter

conv = OmeZarrConverter(
    source='image.tiff',
    destination='out_dir'
)
conv.convert()
```

#### Advanced usage: full control

```python
from bioio_conversion.converters import OmeZarrConverter
from zarr.codecs import BloscCodec

conv = OmeZarrConverter(
    source='multi_scene.czi',
    destination='zarr_output',
    scenes=None,
    name='experiment1',
    tbatch=2,
    num_levels=3,
    downsample_z=True,
    chunk_shape=(1,1,16,256,256),
    shard_shape=(1,1,128,1024,1024),
    memory_target=32*1024*1024,
    dtype='uint16',
    compressor=BloscCodec(),
    zarr_format=3,
)
conv.convert()
```

#### Explicit `level_shapes`

```python
conv = OmeZarrConverter(
    source="image_tczyx.tif",
    destination="out_tczyx",
    level_shapes=[
        (1, 3, 5, 325, 475),
        (1, 3, 2, 162, 238),
        (1, 3, 1, 81, 119),
    ],
)
conv.convert()
```

#### Channel metadata

```python
from bioio_ome_zarr.writers import Channel

channels = [
    Channel(label="DAPI", color="#0000FF", active=True,
            window={"min":100, "max":2000, "start":200, "end":1200}),
    Channel(label="GFP", color="#00FF00", active=True),
    Channel(label="TRITC", color="#FF0000", active=False),
]

conv = OmeZarrConverter(
    source="multi_channel.czi",
    destination="out_channels",
    channels=channels,
)
conv.convert()
```

#### Axes & physical pixel sizes

```python
conv = OmeZarrConverter(
    source="custom_axes.tif",
    destination="out_axes",
    axes_names=["t","c","z","y","x"],
    axes_types=["time","channel","space","space","space"],
    axes_units=[None, None, "micrometer","micrometer","micrometer"],
    physical_pixel_size=[1.0, 1.0, 0.4, 0.108, 0.108],
)
conv.convert()
```

#### Example with fewer dimensions (3D ZYX)

```python
conv = OmeZarrConverter(
    source="volume_zyx.tif",
    destination="out_zyx",
    num_levels=2,
    downsample_z=True,
)
conv.convert()
```

---

### CSV-driven batch conversion

The CSV file should have a header row that names the job parameters. At minimum, include a `source` column (path to each input image). You may also include per-job overrides for any converter option (e.g. `destination`, `scenes`, `tbatch`, `num_levels`, `downsample_z`, `level_shapes`, `memory_target`, `dtype`, `channel_names`, etc.). Values in each row will be merged with the `default_opts` you passed to `BatchConverter`.

```python
from bioio_conversion import BatchConverter

bc = BatchConverter(
    converter_key='ome-zarr',
    default_opts={
        'destination': 'batch_out',
        'tbatch': 4,
    }
)
jobs = bc.from_csv('jobs.csv')  # parse CSV into job dicts
bc.run_jobs(jobs)
```

### Directory-driven batch conversion

```python
from bioio_conversion import BatchConverter

bc = BatchConverter(default_opts={
    'destination': 'dir_out',
})
jobs = bc.from_directory(
    '/data/images',
    max_depth=2,
    pattern='*.tif'
)
bc.run_jobs(jobs)
```

### List-driven batch conversion

```python
from bioio_conversion import BatchConverter

paths = ['/data/a.czi', '/data/b.czi', '/data/c.zarr']
bc = BatchConverter(default_opts={
    'destination': 'list_out',
    'scenes': 0
})
jobs = bc.from_list(paths)
bc.run_jobs(jobs)
```

---

## Command-Line Interface

### `bioio-convert` – single-file conversion

Convert a single image file to OME-Zarr using the configured backend
(default: OME-Zarr).

```bash
bioio-convert SOURCE -d DESTINATION [options]
```

`SOURCE` is the input image file (e.g. `.czi`, `.ome.tiff`, `.nd2`).

**Core options**

* `source` (positional): input image path
* `-d`, `--destination`: output directory for `.ome.zarr` (required)
* `-n`, `--name`: base name for the output (defaults to a value derived from
  the input)
* `-s`, `--scenes`: scene(s) to export (e.g. `0` or `0,2`). If omitted, the
  converter/writer default is used (“all scenes”).
* `--tbatch`: number of timepoints per write batch.
* `--start-t-src`: source T index at which to begin reading (0-based).
* `--start-t-dest`: destination T index at which to begin writing (0-based).

**Multiscale (pyramid) options**

* `--level-shapes`: semicolon-separated per-level shapes (level 0 first).
  Each tuple must have one integer per axis.

  * Example:
    `--level-shapes "1,3,5,512,512;1,3,5,256,256;1,3,5,128,128"`
* `--num-levels`: total number of pyramid levels (including level 0). If
  provided (and `--level-shapes` is not), a half-pyramid is built in X/Y
  (and optionally Z).
* `--downsample-z`: when used with `--num-levels`, also halves the Z
  dimension at each level if a Z axis exists.

**Chunking / sharding (advanced)**

* `--chunk-shape`: single chunk shape tuple applied to all levels
  (e.g. `1,1,16,256,256`).
* `--chunk-shape-per-level`: semicolon-separated chunk shapes per level.
  Overrides `--chunk-shape` and `--memory-target`.
* `--memory-target`: approximate in-memory byte budget used to derive
  per-level chunk shapes when explicit chunk shapes are not provided.
  Defaults to 16 MiB if unset.
* `--shard-shape`: single shard shape tuple for Zarr v3
  (e.g. `1,1,128,1024,1024`).
* `--shard-shape-per-level`: semicolon-separated shard shapes per level
  (Zarr v3). Overrides `--shard-shape`.

When no explicit chunk/shard shapes are given for a Zarr v3 store, the
converter derives them automatically with an uncompressed budget of
**16 MiB per chunk** and **4 GiB per level-0 shard**.

**Writer / metadata options**

* `--dtype`: output dtype override (e.g. `uint16`, `float32`). If omitted,
  the reader’s native dtype is used.
* `--physical-pixel-sizes`: comma-separated floats (one per axis, level 0
  only). Example for `(t,c,z,y,x)`:
  `--physical-pixel-sizes 1.0,1.0,0.4,0.108,0.108`
* `--zarr-format`: target Zarr version:

  * `2` ≈ NGFF 0.4
  * `3` ≈ NGFF 0.5
    If omitted, the writer’s default is used (`3` ≈ NGFF 0.5).

**Channel display options**

These only take effect when `--channel-labels` is provided. All lists must
align by channel index.

* `--channel-labels`: comma-separated channel names
  (e.g. `DAPI,GFP,TRITC`).
* `--channel-colors`: comma-separated colors (CSS color names or hex codes).
  Example: `"#0000FF,#00FF00,#FF0000"`.
* `--channel-actives`: booleans for channel visibility
  (e.g. `true,true,false`).
* `--channel-coefficients`: per-channel intensity coefficients
  (e.g. `1,0.8,1.2`).
* `--channel-families`: intensity family names per channel
  (e.g. `linear,sRGB,sRGB`).
* `--channel-inverted`: booleans for inverted display per channel.
* `--channel-window-min`, `--channel-window-max`,
  `--channel-window-start`, `--channel-window-end`: per-channel windowing
  values. Only used when any window value is provided.

**Axis metadata options**

* `--axes-names`: comma-separated axis names in native axis order.
  Example: `t,c,z,y,x`.
* `--axes-types`: comma-separated axis semantic types
  (e.g. `time,channel,space,space,space`).
* `--axes-units`: comma-separated axis units, in the same order as
  `--axes-names`. Use `none`, `null`, or a blank position for missing units.
  Example for `(t,c,z,y,x)`:
  `s,,um,um,um`.

### `bioio-convert` examples

**Basic usage**

```bash
bioio-convert image.tif -d out_dir
```

**Custom name**

```bash
bioio-convert sample.czi -d out_dir -n my_run
```

**Export all scenes**

```bash
bioio-convert multi_scene.ome.tiff -d zarr_out
```

**Export specific scenes**

```bash
bioio-convert multi_scene.ome.tiff -d zarr_out -s 0,2
```

**Simple half-pyramid (XY only)**

```bash
bioio-convert volume.tif -d out_xy --num-levels 3
```

**Simple half-pyramid (XYZ)**

```bash
bioio-convert volume_tczyx.tif -d out_xyz --num-levels 3 --downsample-z
```

**Explicit level shapes**

```bash
bioio-convert image.tif -d out_explicit \
  --level-shapes "1,3,5,325,475;1,3,2,162,238;1,3,1,81,119"
```

**Dtype and chunking**

```bash
bioio-convert image.tif -d out_dir \
  --dtype uint16 \
  --memory-target 33554432
```

**Custom channels**

```bash
bioio-convert image_with_channels.czi -d out_dir \
  --channel-labels DAPI,GFP,TRITC \
  --channel-colors "#0000FF,#00FF00,#FF0000" \
  --channel-actives true,true,false
```

**Axis metadata**

```bash
bioio-convert image_tczyx.tif -d out_axes \
  --axes-names t,c,z,y,x \
  --axes-types time,channel,space,space,space \
  --axes-units s,,um,um,um
```

**Physical pixel sizes**

```bash
bioio-convert image.tif -d out_dir \
  --physical-pixel-sizes 1.0,1.0,0.4,0.108,0.108
```

---

### `bioio-batch-convert` – batch conversion

Batch mode: convert many files via CSV, directory walk, or an explicit list
of paths. All of the shared OME-Zarr options listed for `bioio-convert`
(`--num-levels`, `--chunk-shape`, `--channel-*`, axis metadata, etc.) are
also accepted here and act as defaults for every job.

```bash
bioio-batch-convert --mode [csv|dir|list] [mode options] [shared options]
```

**Mode selection**

* `-m`, `--mode [csv|dir|list]` (required):

  * `csv`: read jobs from a CSV file.
  * `dir`: scan a directory tree for input files.
  * `list`: use an explicit list of paths from the command line.

**Mode-specific options**

* CSV mode (`--mode csv`)

  * `--csv-file`: path to a CSV describing jobs (one row per job). Each
    column name maps to an `OmeZarrConverter` init argument (e.g. `source`,
    `destination`, `scenes`, `tbatch`, etc.). Values are parsed by the batch
    loader; per-row values override shared defaults from the CLI.

* Directory mode (`--mode dir`)

  * `--directory` / `--dir`: root directory to scan.
  * `--depth`: maximum recursion depth (0 = only top-level files).
  * `--pattern`: glob pattern used when scanning (e.g. "*.czi").

* List mode (`--mode list`)

  * `--paths`: explicit input file paths (repeatable).

**Shared conversion options**

All of the `bioio-convert` options (destination, multiscale, chunking,
channels, axes, etc.) can be passed to `bioio-batch-convert`. They are
converted via `build_ome_zarr_init_opts(...)` and applied as defaults to
every job created by the `BatchConverter`. CSV columns that match a given
argument override the shared defaults on a per-job basis.

### `bioio-batch-convert` examples

**CSV mode**

```bash
bioio-batch-convert \
  --mode csv \
  --csv-file jobs.csv \
  --destination batch_out \
  --tbatch 4 \
  --dtype uint16 \
  --num-levels 3
```

**Directory mode**

```bash
bioio-batch-convert \
  --mode dir \
  --directory data/ \
  --depth 2 \
  --pattern '*.czi' \
  --destination output_zarr \
  --level-shapes "1,3,5,325,475;1,3,2,162,238;1,3,1,81,119"
```

**List mode**

```bash
bioio-batch-convert \
  --mode list \
  --paths a.czi b.czi c.tiff \
  --destination list_out \
  --name batch_run \
  --num-levels 2 \
  --downsample-z
```
---

## License & Issues

BSD 3-Clause [https://bioio-devs.github.io/bioio-conversion/LICENSE](LICENSE)

Report bugs at: [https://github.com/bioio-devs/bioio-conversion/issues](https://github.com/bioio-devs/bioio-conversion/issues)
