# bioio-conversion

[![Build Status](https://github.com/bioio-devs/bioio-conversion/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-conversion/actions)
[![Documentation](https://github.com/bioio-devs/bioio-conversion/actions/workflows/docs.yml/badge.svg)](https://bioio-devs.github.io/bioio-conversion)
[![PyPI version](https://badge.fury.io/py/bioio-conversion.svg)](https://badge.fury.io/py/bioio-conversion)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO conversion tool for going between image formats.

---

## Documentation

See the full documentation on our GitHub Pages site:

[https://bioio-devs.github.io/bioio-conversion](https://bioio-devs.github.io/bioio-conversion)

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
    * Metadata options (`channels`, `axes_names`, `axes_units`, `physical_pixel_size`)
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

## Command-Line Interface: `bioio-convert`

Single-file converter using the configured backend (default: OME-Zarr).

```bash
bioio-convert SOURCE -d DESTINATION [options]
```

**Key options:**

* `source` (positional): input image path
* `-d`, `--destination`: output directory for `.ome.zarr`
* `-n`, `--name`: base name (defaults to source stem)
* `-s`, `--scenes`: scene(s) to export (`0` by default; comma-separated list; `None` = all scenes)
* `--tbatch`: timepoints per write batch (default: `1`)
* `--start-t-src`: source T index to begin reading (default: 0)
* `--start-t-dest`: destination T index to begin writing (default: 0)

**Multiscale:**

* `--level-shapes`: semicolon-separated absolute shapes (level 0 first)
* `--num-levels`: number of pyramid levels (including level 0)
* `--downsample-z`: include Z in half-pyramid downsampling

**Chunking / shards:**

* `--chunk-shape`: explicit chunk shape (e.g. `1,1,16,256,256`)
* `--chunk-shape-per-level`: semicolon-separated chunk shapes per level
* `--memory-target`: bytes per chunk (default: 16 MB)
* `--shard-shape`: shard shape (Zarr v3 only)
* `--shard-shape-per-level`: per-level shard shapes (Zarr v3 only)

**Writer / metadata:**

* `--dtype`: output dtype override (e.g. `uint16`; default: reader’s dtype)
* `--physical-pixel-sizes`: comma-separated floats (per axis, level-0)
* `--zarr-format`: `2` (NGFF 0.4) or `3` (NGFF 0.5); default: writer decides

**Channels:**

* `--channel-labels`: comma-separated channel names
* `--channel-colors`: comma-separated colors (hex or CSS names)
* `--channel-actives`: channel visibility flags (`true,false,...`)
* `--channel-coefficients`: per-channel coefficient floats
* `--channel-families`: intensity family names (`linear,sRGB,...`)
* `--channel-inverted`: channel inversion flags
* `--channel-window-min/max/start/end`: per-channel windowing values

### Examples

#### Basic usage

```bash
bioio-convert image.tif -d out_dir
```

#### Custom name

```bash
bioio-convert sample.czi -d out_dir -n my_run
```

#### Export all scenes

```bash
bioio-convert multi_scene.ome.tiff -d zarr_out
```

#### Export specific scenes

```bash
bioio-convert multi_scene.ome.tiff -d zarr_out -s 0,2
```

#### Simple half-pyramid (XY only)

```bash
bioio-convert volume.tif -d out_xy --num-levels 3
```

#### Simple half-pyramid (XYZ)

```bash
bioio-convert volume_tczyx.tif -d out_xyz --num-levels 3 --downsample-z
```

#### Explicit level shapes

```bash
bioio-convert image.tif -d out_explicit \
  --level-shapes "1,3,5,325,475;1,3,2,162,238;1,3,1,81,119"
```

#### Dtype and chunking

```bash
bioio-convert image.tif -d out_dir --dtype uint16 --memory-target 33554432
```

#### Custom channels

```bash
bioio-convert image_with_channels.czi -d out_dir \
  --channel-labels DAPI,GFP,TRITC \
  --channel-colors "#0000FF,#00FF00,#FF0000" \
  --channel-actives true,true,false
```

#### Physical pixel sizes

```bash
bioio-convert image.tif -d out_dir --physical-pixel-sizes 1.0,1.0,0.4,0.108,0.108
```

---

## Command-Line Interface: `bioio-batch-convert`

Batch mode: convert many files via CSV, directory walk, or explicit list.

```bash
bioio-batch-convert --mode [csv|dir|list] [options]
```

### Examples

#### CSV mode

```bash
bioio-batch-convert \
  --mode csv \
  --csv-file jobs.csv \
  --destination batch_out \
  --tbatch 4 \
  --dtype uint16 \
  --num-levels 3
```

#### Directory mode

```bash
bioio-batch-convert \
  --mode dir \
  --directory data/ \
  --depth 2 \
  --pattern '*.czi' \
  --destination output_zarr \
  --level-shapes "1,3,5,325,475;1,3,2,162,238;1,3,1,81,119"
```

#### List mode

```bash
bioio-batch-convert \
  --mode list \
  --paths a.czi b.czi c.tiff \
  --destination list_out \
  --name batch_run \
  --num-levels 2 --downsample-z
```

---

## License & Issues

BSD 3-Clause [https://bioio-devs.github.io/bioio-conversion/LICENSE](LICENSE)

Report bugs at: [https://github.com/bioio-devs/bioio-conversion/issues](https://github.com/bioio-devs/bioio-conversion/issues)
