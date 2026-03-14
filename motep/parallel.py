from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class DummyMPIComm:
    """Dummy MPI communicator.

    https://github.com/mpi4py/mpi4py/blob/master/src/mpi4py/MPI.pyi
    """

    rank = 0
    size = 1

    def barrier(self) -> None: ...

    def send(self, obj: Any, dest: int, tag: int = 0) -> None: ...

    def recv(self, buf: Any, source: int = 0, tag: int = 0) -> Any:
        return buf

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def gather(self, sendobj: Any, root: int = 0) -> list[Any]:
        return [sendobj]

    def scatter(self, sendobj: Sequence[Any], root: int = 0) -> Any:
        return sendobj[0]

    def allgather(self, sendobj: Any) -> list[Any]:
        return [sendobj]

    def Allreduce(self, sendobj: Any = None, recvobj: Any = None, op=None) -> None:
        recvobj[...] = sendobj[...]

    def allreduce(self, sendobj: Any, op=None) -> Any:
        return sendobj


def _get_world() -> MPI.Comm | DummyMPIComm:
    """Get the world MPI communicator depending on if `mpi4py` is installed.

    Returns
    -------
    MPI.Comm | DummyMPIComm
        World MPI communicator.

    """
    try:
        from mpi4py import MPI
    except ImportError:
        return DummyMPIComm()
    return MPI.COMM_WORLD


world = _get_world()
