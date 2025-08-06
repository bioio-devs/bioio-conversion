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

Converters are Python classes that implement a standard interface for transforming one or more input images into a target format or storage layout.

* **OmeZarrConverter**  
  - **Purpose**: Convert any BioImage-supported input (TIFF, CZI, ND2, etc.) into an OME-Zarr v2 store.  
  - **Features**:  
    - Multi-scene export (`scenes=0`, or a list, None = all scenes)  
    - Multi-resolution pyramids via `xy_scale`, `z_scale`, or explicit `level_scales`  
    - Chunk-size tuning with `chunk_memory_target`  
    - Metadata generation (physical scales, units, channel names, colors)  
  - **Import path**:  
    ```python
    from bioio_conversion.converters import OmeZarrConverter
    ```

* **BatchConverter**  
  - **Purpose**: Orchestrate batch conversions of many files (CSV, directory crawl, or explicit list).  
  - **Features**:  
    - Factory methods: `from_csv()`, `from_directory()`, `from_list()`  
    - Shared `default_opts` for per-job overrides  
    - Dispatch jobs via `.run_jobs()`  
  - **Import path**:  
    ```python
    from bioio_conversion.converters import BatchConverter
    ```

All converter classes live under `bioio_conversion.converters`.

---

### Example: OmeZarrConverter

### Minimal usage

```python
from bioio_conversion.converters import OmeZarrConverter

conv = OmeZarrConverter(
    source='image.tiff',
    destination='out_dir'
)
conv.convert()
```

### Advanced usage: full control

```python
from bioio_conversion.converters import OmeZarrConverter

conv = OmeZarrConverter(
    source='multi_scene.czi',
    destination='zarr_output',
    scenes=None,        # export every scene (default)
    name='experiment1',   # custom base name
    tbatch=2,             # 2 timepoints per write batch
    xy_scale=(0.5, 0.25), # X/Y downsampling levels
    z_scale=(1.0, 0.5),   # Z downsampling levels
    chunk_memory_target=32 * 1024 * 1024,  # 32 MB chunk target
    dtype='uint16',       # output data type override
    channel_names=['DAPI', 'GFP']    # custom labels
)
conv.convert()
```

### CSV-driven batch conversion

The CSV file should have a header row that names the job parameters. At minimum, include a `source` column (path to each input image). You may also include per-job overrides for any converter option (e.g. `destination`, `scenes`, `tbatch`, `xy_scale`, `z_scale`, `chunk_memory_target`, `dtype`, `channel_names`, etc.). Values in each row will be merged with the `default_opts` you passed to `BatchConverter`.

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
* `-s`, `--scenes`: scene(s) to export (`0` by default; comma-separated list for selection; None = all scenes)  
* `--tbatch`: timepoints per write batch (default: `1`)  
* `--level-scales`: TCZYX tuples (e.g. `1,1,1,1,1;1,1,1,0.5,0.5`; default: full resolution only)  
* `--xy-scale`: XY downsampling (e.g. `0.5,0.25`; default: no downsampling)  
* `--z-scale`: Z downsampling (e.g. `1.0,0.5`; default: no downsampling)  
* `--memory-target`: bytes per chunk (default: 16mbs)  
* `--dtype`: output dtype override (e.g. `uint16`; default: reader’s dtype)  
* `--channel-names`: comma-separated labels (default: reader’s channel names or autogenerated)  
* `--auto-dask-cluster`: automatically spin up a local Dask cluster with 8 workers before conversion (default: off) 

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

BSD 3-Clause [https://bioio-devs.github.io/bioio-conversion/LICENSE](LICENSE)

Report bugs at: [https://github.com/bioio-devs/bioio-conversion/issues](https://github.com/bioio-devs/bioio-conversion/issues)
