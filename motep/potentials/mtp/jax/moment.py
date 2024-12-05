import json
import os
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

from .utils import TEST_R_UNITS, TEST_RB_VALUES, make_tensor

DEFAULT_MAX_MOMENTS = 4


#
# Functions for finding moments and all unique contractions for some level
#

# Global dict to store calculated test moments
calculated_test_moments = {}


def _get_test_moments(moments):
    calculated_moments = {}
    for moment in moments:
        mu, nu = moment[0:2]
        m = _get_test_moment(nu, mu)
        calculated_moments[moment] = m
    return calculated_moments


@cache
def _get_test_moment(nu, mu):
    m = _get_test_tensor(nu)
    m = (m * TEST_RB_VALUES[mu]).sum(axis=-1)
    return m


@cache
def _get_test_tensor(nu):
    make_tensor(TEST_R_UNITS, nu)


@cache
def _find_possible_axes(ldim, rdim):
    """Returns possible axes to sum over.

    Returns the allowed axes to sum over (see np.tensordot) of a contraction between ldim and rdim dimensional moments.
    """
    # This is too brute force. ((0, 3), (0, 3), (0, 3), (0, 3)) finally results
    # in almost 5 million possible contractions.
    # Need to reduce this...
    if ldim == 0 or rdim == 0:
        min_naxes = 0
    else:
        min_naxes = 1  # Up to (including) level 20, 0 should not be needed
    max_naxes = np.min([ldim, rdim]) + 1
    l_all_axes = list(range(ldim))
    r_all_axes = list(range(rdim))
    all_axes = []
    for naxes in range(min_naxes, max_naxes):
        # for laxes in permutations(l_all_axes, naxes):
        # for laxes in combinations(l_all_axes, naxes):
        # We always have a symmetric left side moment, so below is enough
        laxes = tuple(l_all_axes[:naxes])
        for raxes in permutations(r_all_axes, naxes):
            axes = (laxes, raxes)
            all_axes.append(axes)
    return all_axes


# @cache  # Slows down
def _get_contraction_dimension(contraction):
    if type(contraction[0]) is not tuple:
        if type(contraction[1]) is tuple:
            raise ValueError()
        return contraction[1]
    ldim = _get_contraction_dimension(contraction[0])
    rdim = _get_contraction_dimension(contraction[1])
    naxes = len(contraction[3][0])
    dim = ldim + rdim - 2 * naxes
    return dim


def _get_cheapest_contraction(map_list):
    lowest_cost = 100_000_000  # Big
    for mapping in map_list:
        cost = 0
        for contraction in _extract_pair_contractions_from_mapping_rec(mapping):
            dim = contraction[2]
            cost += dim  # Resulting dimension... Correct estimate?
        if cost < lowest_cost:
            lowest_cost = cost
            cheapest_mapping = mapping
    return cheapest_mapping


def _flatten_nested_pair_tuples(tpl, lst=None):
    if lst is None:
        lst = []
    if type(tpl[0]) is not tuple:
        if type(tpl[1]) is tuple:
            raise ValueError()
        lst.append(tpl)
        return tuple(lst)
    _flatten_nested_pair_tuples(tpl[0], lst)
    _flatten_nested_pair_tuples(tpl[1], lst)
    return tuple(lst)


class MomentBasis:

    def __init__(
        self,
        max_level: int,
        max_contraction_length: int | None = DEFAULT_MAX_MOMENTS,
    ):
        """
        Parameters
        ---------
        max_level : int
            Defines the maximum level of the moment contractions.

        nmoments_max : int or None
            Sets the upper limit for the number of moments in a contraction.
            Defaults to 4, but can also be None, in which case it is set to
            max_level / 2, i.e. all possible included (Warning, see Note.).

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
        if max_contraction_length is not None:
            self.max_contraction_length = max_contraction_length
        else:
            self.max_contraction_length = int(max_level / 2)

    def init_moment_mappings(self):
        self.scalar_contractions = self.get_moment_contractions()
        self.basic_moments = extract_basic_moments(self.scalar_contractions)
        self.pair_contractions = extract_pair_contractions(self.scalar_contractions)

    def get_moment_contractions(self):
        """Get the contraction list."""
        max_moments = np.min([int(self.max_level / 2), self.max_contraction_length])
        try:
            scalar_contractions = self.read_moments(max_moments)
        except FileNotFoundError:
            scalar_contractions = self.find_moment_contractions(max_moments)
        self.write_moments(scalar_contractions, max_moments)
        return scalar_contractions

    def find_moment_contractions(self, max_moments):
        """Enumerates all possible moments and contractions.

        Returns
        -------
        scalar_contractions : tuple of tuple
            A tuple of tuples representing the moment contractions resulting
            in a unique scalar.

        """
        mu_max = int(np.floor((self.max_level - 2) / 4))
        nu_max = int(np.max([self.max_level / 2 - 2, 0]))
        moment_index_list = list(product(range(mu_max + 1), range(nu_max + 1)))
        scalar_contractions = []
        for nmoments in range(1, max_moments + 1):
            for moment_combo in combinations_with_replacement(
                moment_index_list, nmoments
            ):
                level = np.sum([2 + 4 * mu + nu for mu, nu in moment_combo])
                if level > self.max_level:
                    continue
                possible_contractions = _get_contractions_from_basic_moments(
                    moment_combo
                )
                possible_contractions = [c for c in possible_contractions if c[2] == 0]
                if len(possible_contractions) == 0:
                    continue
                contractions = _extract_unique_contractions(possible_contractions)
                scalar_contractions.extend(contractions)
        scalar_contractions = tuple(scalar_contractions)
        return scalar_contractions

    def read_moments(self, max_number_of_moments):
        filename = _get_filename(self.max_level, max_number_of_moments)
        with open(filename, "r") as f:
            moments = json.load(f)
        moments = _to_tuple_recursively(moments)
        return moments

    def write_moments(self, moments, max_number_of_moments):
        filename = _get_filename(self.max_level, max_number_of_moments)
        with open(filename, "w") as f:
            json.dump(moments, f)


def _get_filename(max_level, max_moments):
    data_path = os.path.dirname(__file__) + "/precomputed_moments"
    if max_moments != np.min([int(max_level / 2), DEFAULT_MAX_MOMENTS]):
        filename = data_path + f"/moments_level{max_level}_max{max_moments}moments.json"
    else:
        filename = data_path + f"/moments_level{max_level}.json"
    return filename


def _to_tuple_recursively(lst):
    return tuple(_to_tuple_recursively(i) if isinstance(i, list) else i for i in lst)


def _get_contractions_from_basic_moments(index_combo):
    # For max 4 moments its okay to semi-explicitly enumerate them.
    moments = [(m[0], m[1], m[1]) for m in index_combo]
    if len(moments) == 1:
        return moments
    elif len(moments) == 2:
        m1, m2 = moments
        contractions = []
        for axes in _find_possible_axes(m1[2], m2[2]):
            dim = _get_contraction_dimension((m1, m2, None, axes))
            contractions.append((m1, m2, dim, axes))
        return contractions
    elif len(moments) == 3:
        first_contractions = []
        for m1, m2 in list(combinations(moments, 2)):
            for axes in _find_possible_axes(m1[2], m2[2]):
                dim = _get_contraction_dimension((m1, m2, None, axes))
                first_contractions.append((m1, m2, dim, axes))
        second_contractions = []
        for contraction1 in first_contractions:
            remaining_moments = deepcopy(moments)
            for m in [contraction1[0], contraction1[1]]:
                remaining_moments.remove(m)
            remaining_moment = remaining_moments[0]
            m1 = remaining_moment
            m2 = contraction1
            contractions2 = []
            for axes in _find_possible_axes(m1[2], m2[2]):
                dim = _get_contraction_dimension((m1, m2, None, axes))
                contractions2.append((m1, m2, dim, axes))
            second_contractions.extend(contractions2)
        return second_contractions
    elif len(moments) == 4:
        first_contractions = []
        for m1, m2 in list(combinations(moments, 2)):
            for axes in _find_possible_axes(m1[2], m2[2]):
                dim = _get_contraction_dimension((m1, m2, None, axes))
                first_contractions.append((m1, m2, dim, axes))
        second_contractions = []
        for contraction1 in first_contractions:
            remaining_moments = deepcopy(moments)
            for m in [contraction1[0], contraction1[1]]:
                remaining_moments.remove(m)
            possible_other_contractions = []
            for m1, m2 in list(combinations(remaining_moments, 2)):
                for axes in _find_possible_axes(m1[2], m2[2]):
                    dim = _get_contraction_dimension((m1, m2, None, axes))
                    possible_other_contractions.append((m1, m2, dim, axes))
            for other in chain(remaining_moments, possible_other_contractions):
                m1 = other
                m2 = contraction1
                contractions2 = []
                for axes in _find_possible_axes(m1[2], m2[2]):
                    dim = _get_contraction_dimension((m1, m2, None, axes))
                    contractions2.append((m1, m2, dim, axes))
                second_contractions.extend(contractions2)
        third_contractions = []
        for contraction2 in second_contractions:
            remaining_moments = deepcopy(moments)
            for m in _flatten_nested_pair_tuples(contraction2):
                remaining_moments.remove(m)
            if len(remaining_moments) == 0:
                third_contractions.append(contraction2)
            else:
                remaining_moment = remaining_moments[0]
                m1 = remaining_moment
                m2 = contraction2
                contractions3 = []
                for axes in _find_possible_axes(m1[2], m2[2]):
                    dim = _get_contraction_dimension((m1, m2, None, axes))
                    contractions3.append((m1, m2, dim, axes))
                third_contractions.extend(contractions3)
        return third_contractions


def _extract_unique_contractions(contractions):
    results = {}
    relative_tolerance = 1e-8
    for contraction in contractions:
        res = float(_test_contraction(contraction))
        for prev_res in results:
            # if np.isclose(res, prev_res, rtol=relative_tolerance):  # Slower
            if np.abs(res - prev_res) / prev_res < relative_tolerance:
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


def _test_contraction(contraction):
    moments = extract_basic_moments([contraction])
    # calculated_moments = _get_test_moments(moments)
    for m, c in _get_test_moments(moments).items():
        calculated_test_moments[m] = c
    pair_contractions = extract_pair_contractions([contraction])
    if len(pair_contractions) == 0:
        return calculated_test_moments[contraction]
    for contraction in pair_contractions:
        if contraction not in calculated_test_moments:
            m1 = calculated_test_moments[contraction[0]]
            m2 = calculated_test_moments[contraction[1]]
            calculated_contraction = np.tensordot(m1, m2, axes=contraction[3])
            calculated_test_moments[contraction] = calculated_contraction
    last_contraction = calculated_test_moments[pair_contractions[-1]]
    return last_contraction


def extract_basic_moments(contractions):
    basic_moments = []
    for contraction in contractions:
        flat = _flatten_nested_pair_tuples(contraction)
        for moment in flat:
            if moment not in basic_moments:
                basic_moments.append(moment)
    return tuple(basic_moments)


def extract_pair_contractions(contractions_list):
    pair_contractions = []
    for contractions in contractions_list:
        lst = _extract_pair_contractions_from_mapping_rec(contractions)
        for contraction in lst:
            if contraction not in pair_contractions:
                pair_contractions.append(contraction)
    return tuple(pair_contractions)


def _extract_pair_contractions_from_mapping_rec(mapping):
    pair_contractions = []
    if type(mapping[0]) is tuple:
        if type(mapping[1]) is not tuple:
            raise ValueError()
        lst1 = _extract_pair_contractions_from_mapping_rec(mapping[0])
        lst2 = _extract_pair_contractions_from_mapping_rec(mapping[1])
        for contraction in chain(lst1, lst2):
            if contraction not in pair_contractions:
                pair_contractions.append(contraction)
        pair_contractions.append(mapping)
    return pair_contractions
