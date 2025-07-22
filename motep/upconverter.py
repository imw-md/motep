"""`motep upconvert` command."""

import argparse
from collections import defaultdict

import numpy as np
from mpi4py import MPI

from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.potentials.mtp.data import MTPData
from motep.setting import load_setting_upconvert


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting", nargs="?")


def _init(src: MTPData, dst: MTPData) -> None:
    """Initialize MTP parameters."""
    dst.version = src.version
    dst.potential_name = src.potential_name
    dst.scaling = src.scaling
    dst.species_count = max(dst.species_count, src.species_count)
    dst.potential_tag = src.potential_tag
    dst.radial_basis_type = src.radial_basis_type
    dst.min_dist = src.min_dist
    dst.max_dist = src.max_dist
    dst.radial_funcs_count = max(dst.radial_funcs_count, src.radial_funcs_count)
    dst.radial_funcs_count = max(dst.radial_funcs_count, src.radial_funcs_count)
    spc = dst.species_count
    rfc = dst.radial_funcs_count
    rbs = dst.radial_basis_size
    dst.radial_coeffs = np.zeros((spc, spc, rfc, rbs))
    dst.species_coeffs = np.zeros(dst.species_count)
    dst.moment_coeffs = np.zeros(dst.alpha_scalar_moments)


def _copy_radial_coeffs(src: MTPData, dst: MTPData) -> None:
    spc = src.species_count
    rfc = src.radial_funcs_count
    rbs = src.radial_basis_size
    dst.radial_coeffs[:spc, :spc, :rfc, :rbs] = src.radial_coeffs

    # Zeros in both radial_coeffs and moment_coeffs may be troublesome during training.
    # Therefore, new radial parts are initialized by the average of the old ones.

    tmp = src.radial_coeffs.mean(axis=0)[None, :, :, :]
    dst.radial_coeffs[spc:, :spc, :rfc, :rbs] = tmp

    tmp = src.radial_coeffs.mean(axis=1)[:, None, :, :]
    dst.radial_coeffs[:spc, spc:, :rfc, :rbs] = tmp

    tmp = src.radial_coeffs.mean(axis=2)[:, :, None, :]
    dst.radial_coeffs[:spc, :spc, rfc:, :rbs] = tmp


def _copy_species_coeffs(src: MTPData, dst: MTPData) -> None:
    dst.species_coeffs[: src.species_count] = src.species_coeffs


def _copy_moment_coeffs(src: MTPData, dst: MTPData) -> None:
    mapping = _get_mapping(src, dst)

    for m0, m1 in enumerate(mapping):
        if m0 not in src.alpha_moment_mapping:
            continue
        if m1 not in dst.alpha_moment_mapping:
            continue
        i0 = np.where(m0 == src.alpha_moment_mapping)[0][0]
        i1 = np.where(m1 == dst.alpha_moment_mapping)[0][0]
        dst.moment_coeffs[i1] = src.moment_coeffs[i0]


def _init_mapping(pot0: MTPData) -> np.ndarray:
    if pot0.alpha_index_times_count == 0:
        m = pot0.alpha_index_basic_count
    else:
        m = max(pot0.alpha_index_times[:, -1]) + 1
    # initialize with the max negative value to raise an error whenever not updated
    return np.full(m, np.iinfo(int).min, dtype=int)


def _get_contractions(pot: MTPData) -> dict[str, np.ndarray]:
    contractions = defaultdict(list)
    for aib in pot.alpha_index_times:
        contractions[aib[3]].append(aib[[0, 1, 2]])
    return {k: np.array(v) for k, v in contractions.items()}


def _get_mapping(src: MTPData, dst: MTPData) -> np.ndarray:
    mapping = _init_mapping(src)

    # mapping of alpha_index_basic
    for i0, aib0 in enumerate(src.alpha_index_basic):
        for i1, aib1 in enumerate(dst.alpha_index_basic):
            if np.all(aib0 == aib1):
                if mapping[i0] != np.iinfo(int).min:
                    raise RuntimeError
                mapping[i0] = i1
                break
        else:
            raise RuntimeError(aib0)

    def _map_contraction(v: np.ndarray) -> np.ndarray:
        v_mapped = np.full_like(v, np.iinfo(int).min)
        v_mapped[:, 0] = mapping[v[:, 0]]
        v_mapped[:, 1] = mapping[v[:, 1]]
        v_mapped[:, 2] = v[:, 2]
        return v_mapped

    def _sort(v: np.ndarray) -> np.ndarray:
        return v[np.lexsort((v[:, 0], v[:, 1]))]

    # mapping of alpha_index_times
    src_contractions = _get_contractions(src)
    dst_contractions = _get_contractions(dst)
    for k0, v0 in src_contractions.items():
        v0_mapped = _map_contraction(v0)
        v0_mapped_sorted = _sort(v0_mapped)
        for k1, v1 in dst_contractions.items():
            v1_sorted = _sort(v1)
            if np.array_equal(v0_mapped_sorted, v1_sorted):
                mapping[k0] = k1
                break
        else:
            raise RuntimeError(k0, v0)

    return mapping


def upconvert(src: MTPData, dst: MTPData) -> None:
    """Upconvert."""
    _init(src, dst)
    _copy_radial_coeffs(src, dst)
    _copy_species_coeffs(src, dst)
    _copy_moment_coeffs(src, dst)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    setting = load_setting_upconvert(args.setting)

    src = read_mtp(setting.potentials.base)
    dst = read_mtp(setting.potentials.initial)

    upconvert(src, dst)

    if rank == 0:
        write_mtp(setting.potentials.final, dst)
