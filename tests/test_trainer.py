import pathlib

from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.mlippy_loss_function import current_value


def test_current_value():
    """Test `current_value`."""
    fn = pathlib.Path(__file__).parent / "test.cfg"
    images = read_cfg(fn, index=":")
    current_value(images, MPI.COMM_WORLD)
