"""Benchmark MPI strong-scaling of the training loss/Jacobian passes.

The loss distributes configurations round-robin across ranks
(``range(rank, ncnf, size)`` in :mod:`motep.loss`) and combines them with
``allreduce``/``Allreduce``. This script measures, per rank count:

* ``loss``    — wall time of a full ``loss(params)`` (E/F/S) evaluation,
* ``jac``     — wall time of a full ``loss_and_jac(params)`` pass (the cost
  that dominates training),
* ``compute`` — the rank-local compute time (the subset loop only), gathered
  across ranks so load imbalance and communication overhead are separable.

``imbal%`` is ``(max - mean) / mean`` of the per-rank compute time: how far the
slowest rank runs ahead of a perfectly balanced split. The gap between the full
``jac`` wall and the slowest rank's ``compute`` is broadcast + reduction +
barrier overhead.

Two ways to run it:

* **Single point** — launch under MPI yourself; it reports one rank count::

      mpirun -n 4 python benchmarks/mpi_scaling.py --level 10 --stride 5

* **Scaling sweep** — run as plain Python with ``--driver``; it re-launches
  itself under ``mpirun -n k`` for each ``k`` and prints a speedup/efficiency
  table (baseline = the smallest rank count)::

      python benchmarks/mpi_scaling.py --driver "1 2 4 8" --level 10 --stride 5

  Pass extra launcher flags with ``=`` so argparse keeps the leading ``--``::

      python benchmarks/mpi_scaling.py --driver "1 2 4" --mpirun-args="--oversubscribe"
"""

import argparse
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import LossFunction, LossSetting
from motep.parallel import world


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=pathlib.Path, default="tests/data_path")
    p.add_argument("--crystal", default="multi")
    p.add_argument("--level", type=int, default=10)
    p.add_argument("--engine", default="cext")
    p.add_argument("--stride", type=int, default=5, help="subsample the training set")
    p.add_argument("--reps", type=int, default=5, help="timed repetitions per pass")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--driver",
        metavar="RANKS",
        default=None,
        help='space-separated rank counts to sweep, e.g. "1 2 4 8" '
        "(runs as plain Python, re-launches under mpirun)",
    )
    p.add_argument("--mpirun", default="mpirun", help="launcher (mpirun, srun, ...)")
    p.add_argument(
        "--mpirun-args",
        default="",
        help='extra launcher args, e.g. "--oversubscribe"',
    )
    return p.parse_args()


def build(args: argparse.Namespace) -> tuple[LossFunction, np.ndarray, int]:
    """Build a fresh, warmed loss with an identical start across ranks."""
    fitting = args.data_path / f"fitting/crystals/{args.crystal}/{args.level:02d}"
    training = args.data_path / f"original/crystals/{args.crystal}/training.cfg"
    if not (fitting / "initial.mtp").exists():
        raise SystemExit(f"missing {fitting / 'initial.mtp'}")

    images = read_cfg(training, index=":")[:: args.stride]
    mtp_data = read_mtp(fitting / "initial.mtp")
    setting = LossSetting(energy_weight=1.0, forces_weight=0.01, stress_weight=0.001)
    loss = LossFunction(
        images, mtp_data=mtp_data, setting=setting, comm=world, engine=args.engine
    )
    # Every rank must start from the same parameters; seed is fixed.
    mtp_data.initialize(rng=np.random.default_rng(args.seed))
    params = np.asarray(mtp_data.parameters, dtype=float)

    # Warm both engine paths (geometry caches, basis-array allocation, JIT).
    loss.loss_and_jac(params)
    loss(params)
    return loss, params, len(images)


def _timed_compute(loss: LossFunction) -> list[float]:
    """Wrap the local passes to accumulate this rank's compute time."""
    acc = [0.0]
    run, jac = loss._run_calculations, loss._run_jac_calculations

    def wrap(fn):
        def wrapper():
            t0 = time.perf_counter()
            fn()
            acc[0] += time.perf_counter() - t0

        return wrapper

    loss._run_calculations = wrap(run)
    loss._run_jac_calculations = wrap(jac)
    return acc


def _time_pass(comm, fn, reps: int, acc: list[float]) -> tuple[float, float]:
    """Return (median full-call wall, this rank's mean local-compute time).

    Barriers bracket each call so the rank-0 wall reflects the slowest rank.
    """
    walls: list[float] = []
    acc[0] = 0.0
    for _ in range(reps):
        comm.barrier()
        t0 = time.perf_counter()
        fn()
        comm.barrier()
        walls.append(time.perf_counter() - t0)
    return statistics.median(walls), acc[0] / reps


def run_worker(args: argparse.Namespace) -> None:
    """Run one rank-count point and print a parseable ``RESULT`` line."""
    comm = world
    loss, params, n_img = build(args)
    acc = _timed_compute(loss)

    loss_ms, _ = _time_pass(comm, lambda: loss(params), args.reps, acc)
    jac_ms, compute_s = _time_pass(
        comm, lambda: loss.loss_and_jac(params), args.reps, acc
    )

    # Gather each rank's local compute time to separate imbalance from comm.
    computes = comm.gather(compute_s, root=0)
    if comm.rank != 0:
        return

    computes = np.asarray(computes) * 1e3
    imbal = 100.0 * (computes.max() - computes.mean()) / max(computes.mean(), 1e-12)
    print(
        f"RESULT ranks={comm.size} images={n_img} reps={args.reps} "
        f"loss_ms={loss_ms * 1e3:.3f} jac_ms={jac_ms * 1e3:.3f} "
        f"compute_ms={computes.max():.3f} imbal_pct={imbal:.1f}",
        flush=True,
    )


def _parse_result(line: str) -> dict:
    fields = {}
    for tok in line.split()[1:]:
        key, _, val = tok.partition("=")
        fields[key] = float(val)
    return fields


def run_driver(args: argparse.Namespace) -> None:
    """Sweep rank counts by re-launching this script under mpirun."""
    ranks = [int(r) for r in args.driver.split()]
    # Forward the worker args verbatim, minus the driver-only options.
    passthru = [
        "--data-path", str(args.data_path),
        "--crystal", args.crystal,
        "--level", str(args.level),
        "--engine", args.engine,
        "--stride", str(args.stride),
        "--reps", str(args.reps),
        "--seed", str(args.seed),
    ]
    extra = args.mpirun_args.split()

    rows: list[dict] = []
    for k in ranks:
        cmd = [args.mpirun, *extra, "-n", str(k), sys.executable, __file__, *passthru]
        print(f"# {' '.join(cmd)}", flush=True)
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        result = next(
            (ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")), None
        )
        if result is None:
            sys.stderr.write(out.stdout + out.stderr)
            raise SystemExit(f"no RESULT from {k} ranks (see output above)")
        rows.append(_parse_result(result))

    base = rows[0]
    print(
        f"\nMPI strong-scaling  crystal={args.crystal} level={args.level} "
        f"engine={args.engine} images={int(base['images'])} reps={args.reps}\n"
    )
    header = (
        f"{'ranks':>5} {'loss_ms':>9} {'jac_ms':>9} {'compute_ms':>11} "
        f"{'imbal%':>7} {'speedup':>8} {'effic%':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        speedup = base["jac_ms"] / max(row["jac_ms"], 1e-12)
        ideal = row["ranks"] / base["ranks"]
        effic = 100.0 * speedup / ideal
        print(
            f"{int(row['ranks']):5d} {row['loss_ms']:9.2f} {row['jac_ms']:9.2f} "
            f"{row['compute_ms']:11.2f} {row['imbal_pct']:7.1f} "
            f"{speedup:7.2f}x {effic:6.1f}%"
        )
    print(
        "\nspeedup/effic are relative to the "
        f"{int(base['ranks'])}-rank baseline (jac pass).",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.driver:
        run_driver(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
