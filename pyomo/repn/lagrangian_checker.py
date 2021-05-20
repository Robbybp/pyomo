#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['LagrangianChecker']

import enum

class LagrangianTerms(enum.Enum):
    OBJECTIVE = 0
    EQUALITY = 1
    INEQUALITY_LOWER = 2
    INEQUALITY_UPPER = 3
    INEQUALITY_LOWER_SLACK = 4
    INEQUALITY_UPPER_SLACK = 5
    PRIMAL_BOUND_UPPER = 6
    PRIMAL_BOUND_LOWER = 7
    SLACK_BOUND_UPPER = 8
    SLACK_BOUND_LOWER = 9


def get_conversion_factors(
        source_factors,
        source_inequality_signs,
        target_factors,
        target_inequality_signs,
        ):
    """
    source_factors and target_factors are factors of each term in the
    Lagrangian function. Theoretically, these factors can be anything
    nonzero. They are typically chosen as +1 for the objective and
    +/- 1 for the constraint/multiplier terms, depending on convention.

    source_inequality_signs and target_inequality_signs contain the signs
    of the factors representing the body (variable part) of the
    inequalities. These differ depending on whether inequalities (and bounds)
    are reformulated into ">= 0" or "<= 0" inequalities by the solver.
    """
    if any(t1 != t2 for t1, t2 in zip(source_factors, target_factors)):
        raise ValueError(
            "Cannot convert multipliers if Lagrangian functions\n"
            "do not contain the same terms."
            )

    LT = LagrangianTerms
    source_objective_factor = source_factors[LT.OBJECTIVE]
    target_objective_factor = target_factors[LT.OBJECTIVE]
    conversion_factors = {}



class LagrangianChecker(object):

    def __init__(model,
            equality_multiplier_suffix=None,
            upper_multiplier_suffix=None,
            lower_multiplier_suffix=None,
            ):
        self._model = model
        self._equality_suffix = equality_multiplier_suffix
        self._upper_suffix = upper_multiplier_suffix
        self._lower_suffix = lower_multiplier_suffix

