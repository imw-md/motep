"""Tests for the MPI-parallel code paths in :mod:`motep.loss`.

These tests exercise the multi-rank behaviour without launching ``mpirun``.
A :class:`FakeComm` models a *buffered* point-to-point communicator: a
"worker" rank pushes tagged messages into a shared store and the root rank
later drains them. That is exactly the access pattern of
:meth:`motep.loss.LossFunction.gather_data`, so it can be driven serially in
a single test process.

"""

from collections import deque
from types import SimpleNamespace

from motep.loss import LossFunction

_MISSING = object()


class FakeComm:
    """Buffered point-to-point communicator sharing one message store.

    Two instances built with the same ``store`` (one per simulated rank)
    exchange messages: :meth:`send` appends to the per-tag queue and
    :meth:`recv` pops from it. Tags are unique per ``(image, message-kind)``
    in ``gather_data``, so cross-rank ordering is irrelevant.
    """

    def __init__(self, rank: int, size: int, store: dict[int, deque]) -> None:
        self.rank = rank
        self.size = size
        self._store = store

    def send(self, obj: object, dest: int, tag: int = 0) -> None:
        self._store.setdefault(tag, deque()).append(obj)

    def recv(self, source: int = 0, tag: int = 0) -> object:
        return self._store[tag].popleft()


def _make_engine(mbd, rbd, *, jac_valid, mgrad_valid=_MISSING):
    """Build a minimal engine stand-in with the attributes gather_data reads."""
    engine = SimpleNamespace(mbd=mbd, rbd=rbd, _jac_valid=jac_valid)
    if mgrad_valid is not _MISSING:
        engine._mgrad_valid = mgrad_valid
    return engine


def _make_loss(rank, size, store, engines):
    """Duck-typed LossFunction stand-in carrying only comm + images."""
    images = [SimpleNamespace(calc=SimpleNamespace(engine=e)) for e in engines]
    return SimpleNamespace(comm=FakeComm(rank, size, store), images=images)


def _transfer(*, producer_engines, consumer_engines, size=2):
    """Run gather_data producer-side then root-side over a shared store.

    Returns the (mutated) consumer-side engines.
    """
    store: dict[int, deque] = {}
    producer = _make_loss(rank=1, size=size, store=store, engines=producer_engines)
    consumer = _make_loss(rank=0, size=size, store=store, engines=consumer_engines)
    # Producer (worker) pushes its slice into the store...
    LossFunction.gather_data(producer)
    # ...then the root drains it, overwriting its placeholders.
    LossFunction.gather_data(consumer)
    return consumer.images


def test_gather_data_transfers_basis_and_stale_flags() -> None:
    """A stale (energy-only) worker result must land on root as stale."""
    prod_mbd, prod_rbd = object(), object()
    # image 0 -> root 0 (never transferred); image 1 -> root 1 (worker slice).
    producer = [
        _make_engine(None, None, jac_valid=False, mgrad_valid=False),
        _make_engine(prod_mbd, prod_rbd, jac_valid=False, mgrad_valid=False),
    ]
    # Root's placeholder for image 1 is wrongly "valid" before the gather.
    consumer = [
        _make_engine(None, None, jac_valid=False, mgrad_valid=False),
        _make_engine("stale", "stale", jac_valid=True, mgrad_valid=True),
    ]

    images = _transfer(producer_engines=producer, consumer_engines=consumer)
    engine = images[1].calc.engine

    assert engine.mbd is prod_mbd
    assert engine.rbd is prod_rbd
    assert engine._jac_valid is False
    assert engine._mgrad_valid is False


def test_gather_data_transfers_valid_flags() -> None:
    """A full jac(mgrad=True) worker result round-trips as valid."""
    prod_mbd, prod_rbd = object(), object()
    producer = [
        _make_engine(None, None, jac_valid=True, mgrad_valid=True),
        _make_engine(prod_mbd, prod_rbd, jac_valid=True, mgrad_valid=True),
    ]
    consumer = [
        _make_engine(None, None, jac_valid=False, mgrad_valid=False),
        _make_engine("stale", "stale", jac_valid=False, mgrad_valid=False),
    ]

    images = _transfer(producer_engines=producer, consumer_engines=consumer)
    engine = images[1].calc.engine

    assert engine.mbd is prod_mbd
    assert engine.rbd is prod_rbd
    assert engine._jac_valid is True
    assert engine._mgrad_valid is True


def test_gather_data_nonmagnetic_engine_has_no_mgrad_flag() -> None:
    """Non-magnetic engines lack ``_mgrad_valid``; it must not be invented."""
    prod_mbd, prod_rbd = object(), object()
    producer = [
        _make_engine(None, None, jac_valid=True),
        _make_engine(prod_mbd, prod_rbd, jac_valid=True),
    ]
    consumer = [
        _make_engine(None, None, jac_valid=False),
        _make_engine("stale", "stale", jac_valid=False),
    ]

    images = _transfer(producer_engines=producer, consumer_engines=consumer)
    engine = images[1].calc.engine

    assert engine.mbd is prod_mbd
    assert engine.rbd is prod_rbd
    assert engine._jac_valid is True
    assert not hasattr(engine, "_mgrad_valid")


def test_gather_data_size_one_is_noop() -> None:
    """At size 1 every image is root-owned; nothing is sent or received."""
    store: dict[int, deque] = {}
    engines = [_make_engine("a", "b", jac_valid=True, mgrad_valid=True)]
    loss = _make_loss(rank=0, size=1, store=store, engines=engines)

    LossFunction.gather_data(loss)

    assert store == {}
    engine = loss.images[0].calc.engine
    assert engine.mbd == "a"
    assert engine._jac_valid is True
