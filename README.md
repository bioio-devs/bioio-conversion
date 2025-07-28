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
pip install bioio-conversion bioio bioio-ome-tiff bioio-ome-zarr
```

---

## Available Converters

* **OmeZarrConverter** – convert to OME-Zarr via BioIO and `bioio-ome-zarr` writer.

All converters live under `bioio_conversion.converters`:

```python
from bioio_conversion.converters import OmeZarrConverter, BatchConverter
```

---

## Python API: OmeZarrConverter

### Minimal usage

```python
from bioio_conversion.converters import OmeZarrConverter

enconv = OmeZarrConverter(
    source='image.tiff',
    destination='out_dir'
)
enconv.convert()
```

### Advanced usage: full control

```python
from bioio_conversion.converters import OmeZarrConverter

conv = OmeZarrConverter(
    source='multi_scene.czi',
    destination='zarr_output',
    scenes=-1,        # export every scene
    name='experiment1',   # custom base name
    overwrite=True,       # remove existing store
    tbatch=2,             # 2 timepoints per write batch
    xy_scale=(0.5, 0.25), # X/Y downsampling levels
    z_scale=(1.0, 0.5),   # Z downsampling levels
    memory_target=32 * 1024 * 1024,  # 32 MB chunk target
    dtype='uint16',       # output data type override
    channel_names=['DAPI', 'GFP']    # custom labels
)
conv.convert()
```

---

## Command-Line Interface: `bioio-convert`

Single-file converter using the configured backend (default: OME-Zarr).

```bash
bioio-convert SOURCE -d DESTINATION [options]
```

### Examples

#### Basic usage

```bash
bioio-convert image.tif -d out_dir
```

#### Overwrite and custom name

```bash
bioio-convert sample.czi -d out_dir -n my_run --overwrite
```

#### Export all scenes

```bash
bioio-convert multi_scene.ome.tiff -d zarr_out --scene "all"
```

#### Export specific scenes

```bash
bioio-convert multi_scene.ome.tiff -d zarr_out --scene 0,2
```

#### XY-only scaling

```bash
bioio-convert image.tif -d out_dir --xy-scale 0.5,0.25
```

#### Z-only scaling

```bash
bioio-convert image.tif -d out_dir --z-scale 1.0,0.5
```

#### Multiscale levels

```bash
bioio-convert image.tif -d out_dir --level-scales "1,1,1,1,1;1,1,1,0.5,0.5"
```

#### Dtype and chunk size override

```bash
bioio-convert image.tif -d out_dir --dtype uint16 --memory-target 33554432
```

#### Custom channel names

```bash
bioio-convert image_with_channels.czi -d out_dir --channel-names DAPI,GFP,TRITC
```

#### Specify converter format

```bash
bioio-convert image.tif -d out_dir --format ome-zarr
```

**Key options:**

* `source` (positional): input image path
* `-d`, `--destination`: output directory for `.ome.zarr`
* `-n`, `--name`: base name (defaults to source stem)
* `-f`, `--format`: converter to use (`ome-zarr`)
* `-s`, `--scenes`: scene(s) to export (`all`, `0`, `0,2`)
* `--overwrite`/`--no-overwrite`: toggle overwrite
* `--tbatch`: timepoints per write batch
* `--level-scales`: TCZYX tuples (e.g. `1,1,1,1,1;1,1,1,0.5,0.5`)
* `--xy-scale`: XY downsampling (e.g. `0.5,0.25`)
* `--z-scale`: Z downsampling (e.g. `1.0,0.5`)
* `--memory-target`: bytes per chunk (default 16777216)
* `--dtype`: output dtype override (e.g. `uint16`)
* `--channel-names`: comma-separated labels

---

## Python API: BatchConverter

### CSV-driven batch conversion

```python
from bioio_conversion import BatchConverter

bc = BatchConverter(
    converter_key='ome-zarr',
    default_opts={
        'destination': 'batch_out',
        'tbatch': 4,
        'overwrite': True
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
    'overwrite': False
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
  --overwrite \
  --dtype uint16 \
  --xy-scale 0.5,0.25
```

#### Directory mode

```bash
bioio-batch-convert \
  --mode dir \
  --directory data/ \
  --depth 2 \
  --pattern '*.czi' \
  --destination output_zarr \
  --scenes all \
  --level-scales "1,1,1,1,1;1,1,1,0.5,0.5"
```

#### List mode

```bash
bioio-batch-convert \
  --mode list \
  --paths a.czi b.czi c.tiff \
  --destination list_out \
  --name batch_run \
  --xy-scale 0.25,0.125
```

---

## License & Issues

MIT License.
Report bugs at: [https://github.com/bioio-devs/bioio-conversion/issues](https://github.com/bioio-devs/bioio-conversion/issues)
