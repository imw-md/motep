"""Utility functions and classes for moment basis representation and creation.

This module provides:
- MomentBasis: Representation of moment basis;
- extract_basic_moments: Extracts the basic moments given a list of contractions;
- extract_pair_contractions: Extracts a list of all pair contractions.
"""

import json
import pathlib
from copy import deepcopy
from functools import cache
from itertools import (
    chain,
    combinations,
    combinations_with_replacement,
    permutations,
    product,
)

import numpy as np
import numpy.typing as npt

from .utils import TEST_R_UNITS, TEST_RB_VALUES, make_tensor

DEFAULT_MAX_MOMENTS = 4
DEFAULT_MAX_MU = 5
DEFAULT_MAX_NU = 10


#
# Functions for finding moments and all unique contractions for some level
#


def _get_test_moments(moments: list) -> dict:
    calculated_moments = {}
    for moment in moments:
        mu, nu = moment[0:2]
        m = _get_test_moment(nu, mu)
        calculated_moments[moment] = m
    return calculated_moments


@cache
def _get_test_moment(nu: int, mu: int) -> npt.NDArray[np.float64]:
    m = _get_test_tensor(nu)
    return (m.T * TEST_RB_VALUES[mu, :]).sum(axis=-1)


@cache
def _get_test_tensor(nu: int) -> npt.NDArray[np.float64]:
    return make_tensor(TEST_R_UNITS, nu)


@cache
def _find_possible_axes(ldim: int, rdim: int) -> list:
    """Find possible axes to sum over.

    Returns
    -------
    The allowed axes to sum over (see np.tensordot) of a contraction between
    ldim and rdim dimensional moments.

    """
    # This is too brute force. ((0, 3), (0, 3), (0, 3), (0, 3)) finally results
    # in almost 5 million possible contractions. Needs to be reduced...
    # Up to (including) level 20, we can exclude 0
    min_naxes = 0 if ldim == 0 or rdim == 0 else 1
    max_naxes = np.min([ldim, rdim]) + 1

    l_all_axes = list(range(ldim))
    r_all_axes = list(range(rdim))

    all_axes = []
    for naxes in range(min_naxes, max_naxes):
        # We always have a symmetric left side moment, so the below combinations
        # are enough
        laxes = tuple(l_all_axes[:naxes])
        for raxes in permutations(r_all_axes, naxes):
            axes = (laxes, raxes)
            all_axes.append(axes)
    return all_axes


# @cache  # Slows down
def _get_contraction_dimension(contraction: list[tuple]) -> int:
    if type(contraction[0]) is not tuple:
        if type(contraction[1]) is tuple:
            raise TypeError
        return contraction[1]
    ldim = _get_contraction_dimension(contraction[0])
    rdim = _get_contraction_dimension(contraction[1])
    naxes = len(contraction[3][0])
    return ldim + rdim - 2 * naxes


def _get_cheapest_contraction(contractions: list) -> list[list[int | list[int]]]:
    lowest_cost = 100_000_000  # Start with something big
    for contraction_tree in contractions:
        cost = 0
        for contraction in _extract_pair_contractions(contraction_tree):
            dim = contraction[2]
            cost += dim  # Resulting dimension... Correct estimate?
        if cost < lowest_cost:
            lowest_cost = cost
            cheapest = contraction_tree
    return cheapest


def _flatten_to_moments(tpl: tuple, lst: list | None = None) -> tuple:
    if lst is None:
        lst = []
    if type(tpl[0]) is not tuple:
        if type(tpl[1]) is tuple:
            raise ValueError()
        lst.append(tpl)
        return tuple(lst)
    _flatten_to_moments(tpl[0], lst)
    _flatten_to_moments(tpl[1], lst)
    return tuple(lst)


class MomentBasis:

    def __init__(
        self,
        max_level: int,
        max_contraction_length: int | None = DEFAULT_MAX_MOMENTS,
        max_mu: int | None = DEFAULT_MAX_MU,
        max_nu: int | None = DEFAULT_MAX_NU,
    ) -> None:
        """Representation of moment basis.

        Parameters
        ----------
        max_level : int
            Defines the maximum level of the moment contractions.

        max_contraction_length, max_mu, max_nu : int or None
            Sets the upper limit for the number of moments in a contraction, the
            mu index and the nu index, respectively. Defaults to 4, 5 and 10,
            respectively, and can also be None, in which case all possible
            according to the equation for max level is included (see Notes).
            The attributes are set to the lowest of the given value and the
            highest possible for a certain max_level.

        Notes
        -----
        A high `max_contraction_lenth` can take very long time.

        The 'level' of a moment is definition as `2 + 4 * mu + nu`
        according to [Podryabinkin_JCP_2023_MLIP]_.

        .. [Podryabinkin_JCP_2023_MLIP]
          E. Podryabinkin, K. Garifullin, A. Shapeev, and I. Novikov,
          J. Chem. Phys. 159, (2023).

        """
        self.max_level = max_level
        self.basic_moments = None
        self.pair_contractions = None
        self.scalar_contractions = None

        mcl = max_contraction_length
        max_possible_mcl = int(self.max_level / 2)
        if mcl is not None and mcl < max_possible_mcl:
            self.max_contraction_length = mcl
        else:
            self.max_contraction_length = max_possible_mcl

        max_possible_mu = int(np.floor((self.max_level - 2) / 4))
        if max_mu is not None and max_mu < max_possible_mu:
            self.max_mu = max_mu
        else:
            self.max_mu = max_possible_mu

        max_possible_nu = int(np.max([self.max_level / 2 - 2, 0]))
        if max_nu is not None and max_nu < max_possible_nu:
            self.max_nu = max_nu
        else:
            self.max_nu = max_possible_nu

    def init_moment_mappings(self) -> None:
        """Initialize moment mappings."""
        self.scalar_contractions = self.get_moment_contractions()
        self.basic_moments = extract_basic_moments(self.scalar_contractions)
        self.pair_contractions = extract_pair_contractions(self.scalar_contractions)

    def get_moment_contractions(self) -> None:
        """Get the contraction list."""
        try:
            scalar_contractions = self.read_moments()
        except FileNotFoundError:
            scalar_contractions = self.find_moment_contractions()
        self.write_moments(scalar_contractions)
        return scalar_contractions

    def find_moment_contractions(self) -> tuple:
        """Enumerate all possible moments and contractions.

        Returns
        -------
        scalar_contractions : tuple of tuple
            A tuple of tuples representing the moment contractions resulting
            in a unique scalar.

        """
        max_mu = int(np.min([np.floor((self.max_level - 2) / 4), self.max_mu]))
        max_nu = int(np.min([np.max([self.max_level / 2 - 2, 0]), self.max_nu]))
        max_nmoments = int(np.min([self.max_level / 2, self.max_contraction_length]))
        index_list = list(product(range(max_mu + 1), range(max_nu + 1)))
        scalar_contractions = []
        for nmoments in range(1, max_nmoments + 1):
            for index_combo in combinations_with_replacement(index_list, nmoments):
                level = np.sum([2 + 4 * mu + nu for mu, nu in index_combo])
                if level > self.max_level:
                    continue
                moments = [(m[0], m[1], m[1]) for m in index_combo]
                contractions = _get_contractions_from_moments(moments)
                if len(contractions) == 0:
                    continue
                scalar_contractions.extend(_extract_unique_contractions(contractions))
        return tuple(scalar_contractions)

    def read_moments(self) -> list:
        """Read moment representations from a json file.

        Returns
        -------
        List of the read moments.

        """
        file = _get_file_path(
            self.max_level,
            self.max_mu,
            self.max_nu,
            self.max_contraction_length,
        )
        with file.open() as f:
            moments = json.load(f)
        moments = _to_tuple_recursively(moments)
        return moments

    def write_moments(self, moments: list) -> None:
        file = _get_file_path(
            self.max_level,
            self.max_mu,
            self.max_nu,
            self.max_contraction_length,
        )
        with file.open("w") as f:
            json.dump(moments, f)


def _get_file_path(
    max_level: int,
    max_mu: int,
    max_nu: int,
    max_moments: int,
) -> pathlib.Path:
    data_path = pathlib.Path(__file__).parent / "precomputed_moments"
    filename = f"moments_level{max_level}"
    if max_mu != int(np.min([np.floor((max_level - 2) / 4), DEFAULT_MAX_MU])):
        filename += f"_maxmu{max_mu}"
    if max_nu != int(np.min([np.max([max_level / 2 - 2, 0]), DEFAULT_MAX_NU])):
        filename += f"_maxnu{max_nu}"
    if max_moments != int(np.min([max_level / 2, DEFAULT_MAX_MOMENTS])):
        filename += f"_max{max_moments}moments"
    return data_path / (filename + ".json")


def _to_tuple_recursively(lst: list) -> tuple:
    return tuple(_to_tuple_recursively(i) if isinstance(i, list) else i for i in lst)


def _get_remaining_moments(
    moments: list[tuple],
    contraction: list[tuple],
) -> list[tuple]:
    remaining_moments = deepcopy(moments)
    for m in _flatten_to_moments(contraction):
        remaining_moments.remove(m)
    return remaining_moments


def _generate_contractions(m1: tuple, m2: tuple) -> list:
    """Generate contractions between two moments/pairs.

    Returns
    -------
    A list of all possible contractions between `m1` and `m2`.

    """
    contractions = []
    for axes in _find_possible_axes(m1[2], m2[2]):
        dim = _get_contraction_dimension((m1, m2, None, axes))
        contractions.append((m1, m2, dim, axes))
    return contractions


def _extend_contractions(current_node: tuple, node_pool: list) -> list:
    """Extend a partially built contraction list.

    Extend a partially built contraction list by combining `current_node` with
    each possible continuation from `node_pool`.

    The extension contains two main points:
      1. Attach each contraction with a raw node from the pool, until
      `node_pool` is exhausted.
      2. Attach each possible recursively contracted tree from the pool.

    Returns
    -------
    A list of contractions.

    """
    # Base case:
    if len(node_pool) == 0:
        return [current_node]

    # Prepare a temporary list with 1) raw nodes
    tmp_node_pool = list(node_pool)
    # and 2) recursively contracted trees
    if len(node_pool) > 1:
        tmp_node_pool.extend(_build_contraction_trees(node_pool))

    # Then contract all of them with the `current_node`, and continue the recursion
    extension = []
    for next_node in tmp_node_pool:
        remaining_nodes = _get_remaining_moments(node_pool, next_node)
        for contraction in _generate_contractions(next_node, current_node):
            extension.extend(_extend_contractions(contraction, remaining_nodes))

    return extension


def _build_contraction_trees(nodes: list) -> list:
    """Build all possible contraction trees, given a list of moments/contractions/nodes.

    Builds the tree with the recursive helper function `_extend_contractions()`.

    Returns
    -------
    A list of contraction trees.

    """
    if len(nodes) == 1:
        return nodes

    contractions = []
    for node_a, node_b in combinations(nodes, 2):
        remaining_nodes = _get_remaining_moments(nodes, [node_a, node_b])
        for initial_contraction in _generate_contractions(node_a, node_b):
            contractions.extend(
                _extend_contractions(initial_contraction, remaining_nodes)
            )

    return contractions


def _get_contractions_from_moments(moments: list[tuple]) -> list:
    """Build a list of all possible contraction trees over the given basic moments.

    Only the resulting scalar contraction trees are returned.

    Returns
    -------
    A list of all possible scaler contraction trees.

    """
    contractions = _build_contraction_trees(moments)
    return [_ for _ in contractions if _[2] == 0]


def _extract_unique_contractions(
    contractions: list[tuple],
    rtol: float = 1e-8,
) -> list[tuple]:
    """Extract the unique contractions from a list by applying a test basis.

    Returns
    -------
    A list of the numerically uniqie contractions.

    """
    results = {}
    for contraction in contractions:
        res = float(_test_contraction(contraction))
        for prev_res in results:  # noqa: PLC0206
            if np.abs(res - prev_res) / prev_res < rtol:
                results[prev_res].append(contraction)
                break
        else:
            results[res] = [contraction]
    unique_contractions = []
    for contraction_list in results.values():
        if len(contraction_list) == 1:
            unique_contractions.append(contraction_list[0])
        else:
            contraction = _get_cheapest_contraction(contraction_list)
            unique_contractions.append(contraction)
    return unique_contractions


# Global dict to store/cache calculated test moments
test_moments_global_cache = {}


def _test_contraction(contraction: tuple) -> float:
    """Apply a fixed test basis to get the numerical value of a contraction.

    Returns
    -------
    A float that numerically represents the contraction for the test basis.

    """
    moments = extract_basic_moments([contraction])
    test_moments_global_cache.update(_get_test_moments(moments))
    pair_contractions = extract_pair_contractions([contraction])
    if len(pair_contractions) == 0:
        return test_moments_global_cache[contraction]
    for pair_contraction in pair_contractions:
        if contraction not in test_moments_global_cache:
            m1 = test_moments_global_cache[pair_contraction[0]]
            m2 = test_moments_global_cache[pair_contraction[1]]
            calculated_contraction = np.tensordot(m1, m2, axes=pair_contraction[3])
            test_moments_global_cache[pair_contraction] = calculated_contraction
    return test_moments_global_cache[pair_contractions[-1]]


def extract_basic_moments(contractions: list[tuple]) -> tuple:
    basic_moments = []
    for contraction_tree in contractions:
        all_moments = _flatten_to_moments(contraction_tree)
        for moment in all_moments:
            if moment not in basic_moments:
                basic_moments.append(moment)
    return tuple(basic_moments)


def extract_pair_contractions(contractions: list[tuple]) -> tuple:
    pair_contractions = []
    for contraction_tree in contractions:
        lst = _extract_pair_contractions(contraction_tree)
        for contraction in lst:
            if contraction not in pair_contractions:
                pair_contractions.append(contraction)
    return tuple(pair_contractions)


def _extract_pair_contractions(contraction_tree: tuple) -> list[tuple]:
    pair_contractions = []
    if type(contraction_tree[0]) is tuple:
        if type(contraction_tree[1]) is not tuple:
            raise ValueError()
        lst1 = _extract_pair_contractions(contraction_tree[0])
        lst2 = _extract_pair_contractions(contraction_tree[1])
        for contraction in chain(lst1, lst2):
            if contraction not in pair_contractions:
                pair_contractions.append(contraction)
        pair_contractions.append(contraction_tree)
    return pair_contractions
