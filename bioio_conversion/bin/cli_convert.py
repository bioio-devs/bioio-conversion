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
def main(source: str, **kwargs: dict[str, Any]) -> None:
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
    """
    try:
        # Map Click params → OmeZarrConverter kwargs
        init_opts = build_ome_zarr_init_opts(**kwargs)

        # Execute conversion
        OmeZarrConverter(source=source, **init_opts).convert()

    except FileExistsError as e:
        # Surface common, expected filesystem issues clearly
        raise click.ClickException(str(e))

    except KeyboardInterrupt:
        # Clean CTRL+C handling
        raise click.Abort()

    except click.ClickException:
        # Preserve Click's formatted errors
        raise

    except Exception as e:
        # Catch-all safety net with clean CLI formatting
        raise click.ClickException(f"Conversion failed: {e}")
