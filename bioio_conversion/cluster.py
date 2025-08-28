import atexit
import signal
from types import FrameType

import psutil
from dask.distributed import Client, LocalCluster


class Cluster:
    """
    A custom Dask cluster class that allows for the creation and management of a Dask
    cluster.
    """

    def __init__(self, n_workers: int = 4) -> None:
        cpu_count = psutil.cpu_count(logical=False) or n_workers
        self._n_workers = max(1, cpu_count // 2)
        self._worker_memory = (
            int(psutil.virtual_memory().available * 0.5) // self._n_workers
        )

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
        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        if client is not None:
            print(client)
            print(client.dashboard_link)
