import abc
import os
from typing import Any, List, Optional


class BaseConverter(abc.ABC):
    """
    Abstract base class defining the interface for any file-format converter.
    """

    def __init__(
        self,
        source: str,
        destination: str,
        *,
        overwrite: bool = False,
        verbose: bool = False,
        name: Optional[str] = None,
        scene: int = 0,
        dtype: Optional[str] = None,
        channel_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.source = source
        self.destination = destination
        self.name = name or os.path.splitext(os.path.basename(source))[0]
        self.overwrite = overwrite
        self.verbose = verbose
        self.scene = scene
        self.dtype_override = dtype
        self.channel_names = channel_names
        self._init_params(**kwargs)

    def log(self, msg: str) -> None:
        """
        Print a message if verbose mode is enabled.
        """
        if self.verbose:
            print(msg)

    @abc.abstractmethod
    def _init_params(self, **kwargs: Any) -> None:
        """
        Initialize converter-specific parameters (e.g., scaling factors).
        """

    @abc.abstractmethod
    def convert(self) -> None:
        """
        Execute the conversion workflow: read from source, process data,
        and write to destination.
        """
