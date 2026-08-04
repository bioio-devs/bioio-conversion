from typing import Any

import click

from ..converters.ome_zarr_converter import OmeZarrConverter
from .ome_zarr_options import build_ome_zarr_init_opts, ome_zarr_options


@click.command()
@click.argument(
    "source",
    type=click.Path(exists=True),
)
@ome_zarr_options(require_destination=True)
def main(source: str, **kwargs: Any) -> None:
    """
    Convert a single image file to OME-Zarr.

    SOURCE is the input image file (for example .czi, .ome.tiff, .nd2).

    All additional flags are shared OME-Zarr conversion options provided via
    ome_zarr_options.

    Examples:

    \b
      # Basic conversion
      bioio-convert input.czi --destination out_dir

    \b
      # With pyramid and chunk control
      bioio-convert input.czi \\
          --destination out_dir \\
          --num-levels 3 \\
          --downsample-z \\
          --chunk-shape 1,1,16,256,256

    \b
      # Recording provenance, with extra kwargs for the metadata reader
      bioio-convert plate.nd2 \\
          --destination out_dir \\
          --include-provenance \\
          --provenance-reader-kwargs '{"plate": "96"}'
    """
    try:
        if (
            kwargs.get("provenance_reader_kwargs") is not None
            and not kwargs["include_provenance"]
        ):
            raise click.UsageError(
                "--provenance-reader-kwargs requires --include-provenance; "
                "without it no provenance is collected and the kwargs would "
                "be silently ignored."
            )

        init_opts = build_ome_zarr_init_opts(**kwargs)
        OmeZarrConverter(source=source, **init_opts).convert()

    except KeyboardInterrupt:
        raise click.Abort()

    except click.ClickException:
        raise

    except Exception as e:
        raise click.ClickException(f"Conversion failed: {e}")
