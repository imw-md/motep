"""Benchmark the ``efs``/``jac`` passes an optimizer performs.

Instruments the loss's ``efs`` (energies/forces/stress) and ``jac`` (parameter
Jacobian) passes during an optimization and reports how many of each the
optimizer triggers, their per-call cost, and the totals.

The jac/efs cost ratio grows strongly with the MTP level, so sweep ``--level``:

    python benchmarks/optimize.py --level 2
    python benchmarks/optimize.py --level 10 --stride 50
    python benchmarks/optimize.py --level 6 --no-jac   # finite-diff, efs-only

With ``jac=True`` (default) BFGS pairs each gradient (``jac``) with line-search
``efs`` evals. With ``--no-jac`` the gradient is finite-differenced, so every
eval is ``efs``.
"""

import argparse
import pathlib
import time

import numpy as np

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import LossFunction, LossSetting
from motep.optimizers import make_optimizer
from motep.parallel import world


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=pathlib.Path, default="tests/data_path")
    p.add_argument("--crystal", default="multi")
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--engine", default="cext")
    p.add_argument("--stride", type=int, default=20, help="subsample the training set")
    p.add_argument("--maxiter", type=int, default=25)
    p.add_argument(
        "--optimizer",
        default="minimize",
        help="make_optimizer name: minimize, DE, DA, LLS, Level2MTP, NI, randomize, GA",
    )
    p.add_argument("--method", default="BFGS", help="scipy method (minimize only)")
    p.add_argument("--no-jac", action="store_true", help="finite-difference gradient")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
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

    # Construct the optimizer first: it sets ``mtp_data.optimized`` to its
    # allowed parameter set, which must be in place before ``initialize``.
    optimizer = make_optimizer(args.optimizer)(loss)
    mtp_data.initialize(rng=np.random.default_rng(args.seed))

    stats = {"efs_n": 0, "efs_t": 0.0, "jac_n": 0, "jac_t": 0.0}
    _run, _jac = loss._run_calculations, loss._run_jac_calculations

    def timed(key: str, fn):
        def wrapper() -> None:
            t0 = time.perf_counter()
            fn()
            stats[f"{key}_t"] += time.perf_counter() - t0
            stats[f"{key}_n"] += 1

        return wrapper

    loss._run_calculations = timed("efs", _run)
    loss._run_jac_calculations = timed("jac", _jac)

    if args.optimizer == "minimize":
        opt_kwargs = {
            "method": args.method,
            "jac": not args.no_jac,
            "options": {"maxiter": args.maxiter},
        }
    else:
        opt_kwargs = {}
    t0 = time.perf_counter()
    optimizer.optimize(**opt_kwargs)
    wall = time.perf_counter() - t0

    efs_avg = stats["efs_t"] / max(stats["efs_n"], 1)
    jac_avg = stats["jac_t"] / max(stats["jac_n"], 1)
    engine = stats["efs_t"] + stats["jac_t"]

    detail = (
        f"method={args.method} jac={not args.no_jac}"
        if args.optimizer == "minimize"
        else ""
    )
    print(
        f"\ncrystal={args.crystal} level={args.level} engine={args.engine} "
        f"images={len(images)} optimizer={args.optimizer} {detail}"
    )
    print(
        f"efs (E/F/S)     : {stats['efs_n']:4d} calls  {efs_avg * 1e3:8.2f} ms/call"
        f"  {stats['efs_t']:8.3f}s"
    )
    print(
        f"jac (d/dparams) : {stats['jac_n']:4d} calls  {jac_avg * 1e3:8.2f} ms/call"
        f"  {stats['jac_t']:8.3f}s"
    )
    if stats["jac_n"]:
        print(f"efs:jac ratio   = {stats['efs_n'] / stats['jac_n']:.2f}")
        print(f"jac/efs cost    = {jac_avg / max(efs_avg, 1e-12):.1f}x")
    print(f"engine total    : {engine:8.3f}s")
    if stats["jac_n"]:
        # each eval served by one jac pass (E/F/S + basis), gradient read free
        print(f"single-pass jac : {stats['efs_n'] * jac_avg:8.3f}s  (for comparison)")
    print(f"optimize() wall : {wall:8.3f}s")


if __name__ == "__main__":
    main()
