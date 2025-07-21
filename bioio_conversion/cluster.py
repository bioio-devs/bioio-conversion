import atexit
import signal
from types import FrameType

import ngff_zarr
from dask.distributed import Client, LocalCluster


class Cluster:
    """
    A custom Dask cluster helper that starts a LocalCluster
    with auto-shutdown on process exit or signals.
    """

    def __init__(self, n_workers: int = 4) -> None:
        self._n_workers = n_workers
        self._worker_memory = ngff_zarr.config.memory_target // self._n_workers
        try:
            import psutil

            cpu_count = psutil.cpu_count(logical=False) or n_workers
            self._n_workers = max(1, cpu_count // 2)
            self._worker_memory = ngff_zarr.config.memory_target // self._n_workers
        except ImportError:
            pass

    def start(self) -> Client:
        """
        Start the Dask LocalCluster and return a Client.
        Registers clean shutdown on exit or SIGINT/SIGTERM.
        """
        cluster = LocalCluster(
            n_workers=self._n_workers,
            memory_limit=self._worker_memory,
            processes=True,
            threads_per_worker=2,
            scheduler_port=0,
        )
        client = Client(cluster)

        def _shutdown(sig: int, frame: FrameType | None = None) -> None:
            client.shutdown()

        atexit.register(client.shutdown)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _shutdown)

        return client
