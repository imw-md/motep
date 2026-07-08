"""Locate the level where motep's parameter-Jacobian pass goes bandwidth bound.

The cext ``train`` engine builds, per atom, a ``moment_jac_rc`` scratch buffer
of size ``n_basic * species * radial_funcs * radial_basis * n_neighbors * 3``
doubles. Once this no longer fits in cache the kernel becomes memory-bandwidth
bound and its per-level cost climbs steeply. This script sweeps levels and times
the motep Jacobian pass (``calc.compute_jacobian``) against a reference to find
that crossover.

Two references are supported:

* ``--mlp-cfg FILE``: times the real ``mlp train`` CLI and motep on the same
  configurations. ``mlp train`` is timed by two-point subtraction -- run at two
  iteration limits and divide the time difference by the iteration delta -- so
  all fixed startup cost cancels, leaving milliseconds per BFGS iteration. Both
  the MLIP-2 (``--max-iter``, ``--trained-pot-name``) and MLIP-3 / spin-MLIP
  (``--iteration_limit``, ``--save_to``) dialects are supported and auto-detected
  from ``mlp help train``; override with ``--mlp-flavor``. The trained potential
  is always written to a temp file (MLIP-3 ``--save_to`` overwrites otherwise).

  One BFGS iteration includes a line search, so it spans *several* EFS+gradient
  evaluations over the set -- not one -- whereas the motep column times a single
  Jacobian pass. Pass ``--mlp-evals-per-iter N`` to divide the mlp time by ``N``
  and compare per-evaluation; the column is then labelled ``mlp/eval``. Note the
  two-point subtraction cancels fixed cost but does not normalise the
  line-search count, so ``N`` is an estimate of its average over iterations
  ``K1..K2``.

* no ``--mlp-cfg``: uses motep ``run`` mode (EFS only, no parameter Jacobian) as
  the reference.

The ``norm`` column is the motep Jacobian time per atom-image per unit of
``n_basic``. It is ~flat while the buffer is cache resident and rises once the
kernel goes bandwidth bound; the level where it starts climbing is the crossover.

Run::

    python -m benchmarks.crossover --mlp-cfg /path/to/subset.cfg \
        --levels 8 10 12 14 16 18 20

Prepare a small subset cfg (keeps the sweep fast).
"""

import argparse
import os
import pathlib
import subprocess
import tempfile
from time import perf_counter

import motep
from motep.calculator import MTP
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp

DATA_PATH = pathlib.Path(motep.__file__).parent.parent / "tests" / "data_path"


def _time(fn, n_repeat: int) -> float:
    """Return the best wall time in milliseconds over ``n_repeat`` runs."""
    best = float("inf")
    for _ in range(n_repeat):
        t0 = perf_counter()
        fn()
        best = min(best, perf_counter() - t0)
    return best * 1000.0


def _moment_jac_rc_bytes(mtp_data, n_neighbors: int) -> int:
    n_basic = len(mtp_data.alpha_index_basic)
    rfc = mtp_data.radial_funcs_count
    rbs = mtp_data.radial_basis.size
    spc = mtp_data.species_count
    return n_basic * spc * rfc * rbs * n_neighbors * 3 * 8


def _make_calc(pot_path: pathlib.Path, species: list[int], *, mode: str) -> MTP:
    mtp_data = read_mtp(pot_path)
    mtp_data.species = species
    return MTP(mtp_data, engine="cext", mode=mode)


def _detect_mlp_flavor(mlp_bin: str) -> str:
    """Return ``"mlip3"`` (spin-MLIP / MLIP-3) or ``"mlip2"`` from the CLI help."""
    text = ""
    for probe in (["help", "train"], ["train", "--help"]):
        try:
            res = subprocess.run(
                [mlp_bin, *probe],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            text += (res.stdout or "") + (res.stderr or "")
        except (OSError, subprocess.SubprocessError):
            pass
    if "iteration_limit" in text:
        return "mlip3"
    if "max-iter" in text:
        return "mlip2"
    msg = (
        f"could not detect the mlp CLI dialect from `{mlp_bin} help train`; "
        "pass --mlp-flavor mlip2|mlip3 explicitly"
    )
    raise RuntimeError(msg)


def _mlp_train_argv(
    flavor: str,
    pot_path: pathlib.Path,
    cfg_path: pathlib.Path,
    max_iter: int,
    out: str,
) -> list[str]:
    """Build the ``train`` argument vector for the given CLI dialect.

    A large iteration budget with a vanishing tolerance forces the optimiser to
    keep taking (full EFS+gradient) steps rather than stopping early, so the two
    iteration limits actually differ in work. ``--save_to`` / ``--trained-pot-name``
    always points at a temp file (MLIP-3 overwrites the input MTP otherwise).
    """
    if flavor == "mlip3":
        return [
            "train",
            str(pot_path),
            str(cfg_path),
            f"--iteration_limit={max_iter}",
            "--skip_preinit=true",
            "--tolerance=1e-16",
            f"--save_to={out}",
            "--log=none",
        ]
    return [
        "train",
        str(pot_path),
        str(cfg_path),
        f"--max-iter={max_iter}",
        "--skip-preinit",
        "--bfgs-conv-tol=1e-16",
        f"--trained-pot-name={out}",
    ]


def _time_mlp_per_iter(
    mlp_bin: str,
    flavor: str,
    pot_path: pathlib.Path,
    cfg_path: pathlib.Path,
    iters: tuple[int, int],
    tmpdir: str,
    evals_per_iter: float | None = None,
) -> float:
    """Milliseconds per BFGS iteration of ``mlp train`` (two-point subtraction).

    Running ``mlp train`` at two iteration limits and differencing removes every
    fixed cost -- potential/cfg loading, the initial rescaling/pre-init phase,
    and the final error report over the set -- leaving the marginal cost of one
    training iteration.

    One such iteration performs a line search, hence several EFS+gradient sweeps
    over the set rather than one. Pass ``evals_per_iter`` to divide by that count
    and return milliseconds per evaluation instead. The subtraction cancels fixed
    cost but not the line-search count, so this is an average over ``K1..K2``.
    """
    out = os.path.join(tmpdir, "trained.mtp")

    def run(max_iter: int) -> float:
        cmd = [mlp_bin, *_mlp_train_argv(flavor, pot_path, cfg_path, max_iter, out)]
        t0 = perf_counter()
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return perf_counter() - t0

    k1, k2 = iters
    t1 = run(k1)
    t2 = run(k2)
    per_iter = (t2 - t1) / (k2 - k1) * 1000.0
    return per_iter if evals_per_iter is None else per_iter / evals_per_iter


def _species_of(images) -> list[int]:
    species: list[int] = []
    for z in images[0].numbers:
        if z not in species:
            species.append(z)
    return species


def main(
    levels: list[int],
    nimages: int,
    n_repeat: int,
    mlp_cfg: str | None,
    mlp_bin: str,
    mlp_iters: tuple[int, int],
    mlp_flavor: str,
    mlp_evals_per_iter: float | None,
) -> None:
    data_path = DATA_PATH
    crystal = "cubic"

    if mlp_cfg is not None:
        cfg_path = pathlib.Path(mlp_cfg)
        images = read_cfg(cfg_path, index=slice(None))
        if mlp_flavor == "auto":
            mlp_flavor = _detect_mlp_flavor(mlp_bin)
        ref_label = "mlp/iter" if mlp_evals_per_iter is None else "mlp/eval"
    else:
        cfg_path = data_path / f"original/crystals/{crystal}/training.cfg"
        images = read_cfg(cfg_path, index=slice(0, nimages))
        ref_label = "run/pass"

    species = _species_of(images)
    natoms = len(images[0])
    ref_desc = f"{ref_label} ({mlp_flavor})" if mlp_cfg is not None else ref_label
    print(f"=== {len(images)} configs x {natoms} atoms | reference = {ref_desc} ===")
    header = (
        f"{'lvl':>3} {'n_basic':>7} {'mjac_rc/atom':>12} "
        f"{'motep(ms)':>10} {ref_label:>10} {'motep/ref':>10} {'norm(us)':>9}"
    )
    print(header)

    tmpdir = tempfile.mkdtemp(prefix="motep_crossover_")

    for level in levels:
        pot_path = data_path / f"fitting/crystals/{crystal}/{level:02d}/pot.mtp"
        if not pot_path.exists():
            continue

        mtp_data = read_mtp(pot_path)
        n_basic = len(mtp_data.alpha_index_basic)

        train_calc = _make_calc(pot_path, species, mode="train")
        train_calc.compute_jacobian(images[-1])  # warm up
        nn = train_calc.engine._neighbors.shape[1]
        mjrc_kb = _moment_jac_rc_bytes(mtp_data, nn) / 1024.0

        def motep_pass(c=train_calc):
            for a in images:
                c.compute_jacobian(a)

        t_motep = _time(motep_pass, n_repeat)
        norm_us = t_motep * 1e3 / (natoms * len(images) * n_basic)

        if mlp_cfg is not None:
            t_ref = _time_mlp_per_iter(
                mlp_bin,
                mlp_flavor,
                pot_path,
                cfg_path,
                mlp_iters,
                tmpdir,
                mlp_evals_per_iter,
            )
        else:
            run_calc = _make_calc(pot_path, species, mode="run")
            run_calc.get_potential_energy(images[-1])

            def run_pass(c=run_calc):
                for a in images:
                    c.get_potential_energy(a)

            t_ref = _time(run_pass, n_repeat)

        print(
            f"{level:>3} {n_basic:>7} {mjrc_kb:>10.1f}K "
            f"{t_motep:>10.1f} {t_ref:>10.1f} {t_motep / t_ref:>10.2f} {norm_us:>9.3f}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[8, 10, 12, 14, 16, 18, 20],
    )
    parser.add_argument(
        "--mlp-cfg",
        default=None,
        help="cfg file to time real `mlp train` against motep on the same configs. "
        "If omitted, motep `run` mode is used as the reference instead.",
    )
    parser.add_argument("--mlp-bin", default="mlp", help="path to the mlp executable")
    parser.add_argument(
        "--mlp-flavor",
        choices=("auto", "mlip2", "mlip3"),
        default="auto",
        help="mlp CLI dialect: mlip2 (--max-iter/--trained-pot-name) or "
        "mlip3/spin-MLIP (--iteration_limit/--save_to). Default auto-detects.",
    )
    parser.add_argument(
        "--mlp-iters",
        nargs=2,
        type=int,
        default=(4, 14),
        metavar=("K1", "K2"),
        help="two --max-iter values differenced to get ms/iteration",
    )
    parser.add_argument(
        "--mlp-evals-per-iter",
        type=float,
        default=None,
        metavar="N",
        help="divide the mlp time by N, the average number of EFS+gradient "
        "evaluations per BFGS iteration (a line search does several). Reports "
        "ms/evaluation instead of ms/iteration, comparable to one motep pass.",
    )
    parser.add_argument(
        "--nimages",
        type=int,
        default=10,
        help="configs to use from the default training set (proxy mode only)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="timed repeats for motep; the best (min) time is reported",
    )
    args = parser.parse_args()
    main(
        args.levels,
        args.nimages,
        args.repeat,
        args.mlp_cfg,
        args.mlp_bin,
        tuple(args.mlp_iters),
        args.mlp_flavor,
        args.mlp_evals_per_iter,
    )
