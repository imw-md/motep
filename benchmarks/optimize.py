"""Benchmark the ``efs``/``jac`` passes an optimizer performs.

For ``--optimizer minimize`` with a gradient it runs a warm A/B of two schemes:

* ``split`` — ``fun`` is an ``efs`` (E/F/S) pass, ``jac`` a separate parameter-
  Jacobian pass.
* ``fused`` — ``fun`` returns ``(loss, jac)`` from a single Jacobian pass
  (SciPy ``jac=True``), so the gradient is free and no redundant ``efs`` runs.

Both paths are warmed first (geometry caches, basis-array allocation, etc), so
the reported per-call costs exclude first-call overhead.

    python benchmarks/optimize.py --level 2
    python benchmarks/optimize.py --level 10 --stride 50

Any other ``--optimizer`` (DE, DA, LLS, ...) is run once and instrumented.
"""

import argparse
import pathlib
import time

import numpy as np
from scipy.optimize import minimize

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import LossFunction, LossSetting
from motep.optimizers import make_optimizer
from motep.parallel import world

_UNBOUNDED = {
    "cg",
    "bfgs",
    "newton-cg",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
}


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


def build(args: argparse.Namespace) -> tuple:
    """Build a fresh, warmed loss + optimizer (identical start across runs)."""
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
    optimizer = make_optimizer(args.optimizer)(loss)  # sets ``mtp_data.optimized``
    mtp_data.initialize(rng=np.random.default_rng(args.seed))

    # Warm up both engine paths: geometry caches, basis-array allocation, JIT.
    loss.loss_and_jac(mtp_data.parameters)
    loss(mtp_data.parameters)
    return loss, optimizer, mtp_data, len(images)


def instrument(loss) -> dict:
    """Wrap the forward/jac passes to count and time them; returns the stats."""
    stats = {"efs_n": 0, "efs_t": 0.0, "jac_n": 0, "jac_t": 0.0}
    run, jac = loss._run_calculations, loss._run_jac_calculations

    def timed(key, fn):
        def wrapper():
            t0 = time.perf_counter()
            fn()
            stats[f"{key}_t"] += time.perf_counter() - t0
            stats[f"{key}_n"] += 1

        return wrapper

    loss._run_calculations = timed("efs", run)
    loss._run_jac_calculations = timed("jac", jac)
    return stats


def run_minimize(args: argparse.Namespace, scheme: str) -> dict:
    """Run ``scipy.minimize`` in the ``split`` or ``fused`` scheme (serial)."""
    loss, _, mtp_data, _ = build(args)
    stats = instrument(loss)
    x0 = mtp_data.parameters
    bounds = None if args.method.lower() in _UNBOUNDED else mtp_data.get_bounds()
    opts = {"maxiter": args.maxiter}
    t0 = time.perf_counter()
    if scheme == "split":
        minimize(
            loss,
            x0,
            jac=loss.jac,
            method=args.method,
            bounds=bounds,
            options=opts,
        )
    else:
        minimize(
            loss.loss_and_jac,
            x0,
            jac=True,
            method=args.method,
            bounds=bounds,
            options=opts,
        )
    stats["wall"] = time.perf_counter() - t0
    return stats


def report(name: str, s: dict) -> None:
    ef = s["efs_t"] / max(s["efs_n"], 1)
    jf = s["jac_t"] / max(s["jac_n"], 1)
    print(f"[{name}]")
    print(f"  efs : {s['efs_n']:3d} calls  {ef * 1e3:7.2f} ms  {s['efs_t']:7.3f}s")
    print(f"  jac : {s['jac_n']:3d} calls  {jf * 1e3:7.2f} ms  {s['jac_t']:7.3f}s")
    print(f"  wall: {s['wall']:7.3f}s")


def main() -> None:
    args = parse_args()
    _, _, _, n_img = build(args)
    print(
        f"\ncrystal={args.crystal} level={args.level} engine={args.engine} "
        f"images={n_img} optimizer={args.optimizer}"
    )

    if args.optimizer == "minimize" and not args.no_jac:
        print(f"method={args.method}  (warm split vs fused)\n")
        split = run_minimize(args, "split")
        fused = run_minimize(args, "fused")
        report("split", split)
        report("fused", fused)
        dw = split["wall"] - fused["wall"]
        print(
            f"\nwall  split - fused : {dw:+.3f}s "
            f"({100 * dw / max(fused['wall'], 1e-12):+.1f}%)"
        )
        return

    loss, optimizer, _, _ = build(args)
    stats = instrument(loss)
    opt_kwargs = (
        {"jac": False, "options": {"maxiter": args.maxiter}}
        if args.optimizer == "minimize"
        else {}
    )
    t0 = time.perf_counter()
    optimizer.optimize(**opt_kwargs)
    stats["wall"] = time.perf_counter() - t0
    print()
    report(args.optimizer, stats)


if __name__ == "__main__":
    main()
