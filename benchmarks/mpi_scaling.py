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

* **Strong-scaling sweep** — run as plain Python with ``--driver``; it
  re-launches itself under ``mpirun -n k`` for each ``k``, keeping the dataset
  fixed, and prints an efficiency table (baseline = the smallest rank count).
  Ideal: the ``jac`` wall shrinks as ``1/ranks``::

      python benchmarks/mpi_scaling.py --driver "1 2 4 8" --level 10 --stride 5

* **Weak-scaling sweep** — add ``--weak``; the dataset grows with the rank
  count so each rank keeps ``--per-rank`` configs. Ideal: the ``jac`` wall stays
  constant as the total work grows::

      python benchmarks/mpi_scaling.py --driver "1 2 4 8" --weak --per-rank 16

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

# Anchor the default dataset to the repo, so the script runs from any CWD.
_DEFAULT_DATA_PATH = (pathlib.Path(__file__).parent / "../tests/data_path").resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=pathlib.Path, default=_DEFAULT_DATA_PATH)
    p.add_argument("--crystal", default="multi")
    p.add_argument("--level", type=int, default=10)
    p.add_argument("--engine", default="cext")
    p.add_argument("--stride", type=int, default=5, help="subsample the training set")
    p.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="tile/truncate the dataset to exactly this many configs "
        "(used by weak scaling; overrides the strided count)",
    )
    p.add_argument("--reps", type=int, default=5, help="timed repetitions per pass")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--driver",
        metavar="RANKS",
        default=None,
        help='space-separated rank counts to sweep, e.g. "1 2 4 8" '
        "(runs as plain Python, re-launches under mpirun)",
    )
    p.add_argument(
        "--weak",
        action="store_true",
        help="weak scaling: hold configs-per-rank fixed (see --per-rank) so the "
        "total dataset grows with the rank count, instead of strong scaling",
    )
    p.add_argument(
        "--per-rank",
        type=int,
        default=16,
        help="configs per rank for --weak scaling",
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
    if args.num_images is not None:
        if not images:
            raise SystemExit("no images to tile")
        # Tile (cycle) the strided pool up to the requested count, then truncate.
        reps = -(-args.num_images // len(images))  # ceil division
        images = (images * reps)[: args.num_images]
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
    """Sweep rank counts by re-launching this script under mpirun.

    Strong scaling holds the dataset fixed; weak scaling (``--weak``) holds
    ``--per-rank`` configs per rank fixed, so the total grows with the ranks.
    """
    ranks = [int(r) for r in args.driver.split()]
    extra = args.mpirun_args.split()

    rows: list[dict] = []
    for k in ranks:
        # Forward the worker args, minus the driver-only options.
        passthru = [
            "--data-path", str(args.data_path),
            "--crystal", args.crystal,
            "--level", str(args.level),
            "--engine", args.engine,
            "--stride", str(args.stride),
            "--reps", str(args.reps),
            "--seed", str(args.seed),
        ]
        if args.weak:
            # Fixed work per rank: total configs scale with the rank count.
            passthru += ["--num-images", str(args.per_rank * k)]
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
    kind = "weak" if args.weak else "strong"
    sizing = (
        f"per_rank={args.per_rank}" if args.weak else f"images={int(base['images'])}"
    )
    print(
        f"\nMPI {kind}-scaling  crystal={args.crystal} level={args.level} "
        f"engine={args.engine} {sizing} reps={args.reps}\n"
    )
    header = (
        f"{'ranks':>5} {'images':>7} {'loss_ms':>9} {'jac_ms':>9} "
        f"{'compute_ms':>11} {'imbal%':>7} {'effic%':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if args.weak:
            # Ideal weak scaling keeps the jac wall constant as work grows.
            effic = 100.0 * base["jac_ms"] / max(row["jac_ms"], 1e-12)
        else:
            # Ideal strong scaling shrinks the jac wall as 1/ranks.
            speedup = base["jac_ms"] / max(row["jac_ms"], 1e-12)
            effic = 100.0 * speedup / (row["ranks"] / base["ranks"])
        print(
            f"{int(row['ranks']):5d} {int(row['images']):7d} "
            f"{row['loss_ms']:9.2f} {row['jac_ms']:9.2f} "
            f"{row['compute_ms']:11.2f} {row['imbal_pct']:7.1f} {effic:6.1f}%"
        )
    ideal = "constant jac wall" if args.weak else "jac wall ~ 1/ranks"
    print(
        f"\neffic% is relative to the {int(base['ranks'])}-rank baseline "
        f"(ideal: {ideal}).",
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
