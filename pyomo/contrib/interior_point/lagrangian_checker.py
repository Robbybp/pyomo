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

from pyomo.core.base.objective import Objective
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_ad
from pyomo.common.collections import ComponentMap
from pyomo.core.kernel.objective import minimize, maximize

class LagrangianTerms(enum.Enum):
    OBJECTIVE = 0
    EQUALITY = 1
    INEQUALITY = 2
    INEQUALITY_LOWER = 3
    INEQUALITY_UPPER = 4
    INEQUALITY_LOWER_SLACK = 5
    INEQUALITY_UPPER_SLACK = 6
    PRIMAL_BOUND_UPPER = 7
    PRIMAL_BOUND_LOWER = 8
    SLACK_BOUND_UPPER = 9
    SLACK_BOUND_LOWER = 10


class InequalityConvention(enum.Enum):
    LESS_THAN_ZERO = 0
    GREATER_THAN_ZERO = 1


class BoundDirection(enum.Enum):
    LOWER = 0
    UPPER = 1


def _check_nonzero(term, factor):
    if factor == 0:
        raise ValueError(
            "Cannot compare Lagrangians because term %s has factor zero.\n"
            "If this term is not present, it should be omitted."
            % term
            )


def _check_contained(term, factor_dict):
    if term not in factor_dict:
        raise ValueError(
            "Term %s does not appear in Lagrangian.\n"
            "Cannot compare formulations if terms are not the same."
            % term
            )


def get_multiplier_conversion_factors(
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

    Returns a dict mapping terms of lagrangian to the factor that should be
    multiplied 
    """
    for term, factor in source_factors.items():
        _check_contained(term, target_factors)
        _check_nonzero(term, factor)
    for term, factor in target_factors.items():
        _check_contained(term, source_factors)
        _check_nonzero(term, factor)

    LT = LagrangianTerms
    OBJ = LT.OBJECTIVE
    objective_factor = target_factors[OBJ]/source_factors[OBJ]\
            if OBJ in target_factors else 1.0

    conversion_factors = {}

    EQ = LT.EQUALITY
    if EQ in source_factors:
        conversion_factors[EQ] = (
                objective_factor*source_factors[EQ]/target_factors[EQ]
                )

    return conversion_factors


class LagrangianChecker(object):

    def __init__(self, model, multiplier_suffix_map):
        self._model = model
        self._multiplier_suffix_map = multiplier_suffix_map

    def _check_compatible_convention(self, convention):
        suffix_map = self._multiplier_suffix_map
        for term in convention:
            if term != LagrangianTerms.OBJECTIVE:
                # Make sure every non-objective term the solver expects
                # has a suffix for the multiplier values.
                #
                # Extra suffixes are fine.
                if term not in suffix_map:
                    raise RuntimeError(
                        "Was not provided a suffix for Lagrangian term %s."
                        % term
                        )

    def get_lagrangian(self, convention, bound_convention=None):
        model = self._model
        suffix_map = self._multiplier_suffix_map
        self._check_compatible_convention(convention)
        LT = LagrangianTerms
        BD = BoundDirection
        IC = InequalityConvention

        term_exprs = []

        objective_list = list(
                model.component_data_objects(Objective, active=True)
                )
        if len(objective_list) == 1:
            # Only infer sense from objective if exactly one objective is
            # provided. Unclear if anything other than this should be
            # supported.
            obj_sense = objective_list[0].sense
        else:
            # Not sure if anything else should be supported...
            raise RuntimeError()

        if obj_sense in bound_convention:
            bound_convention = bound_convention[obj_sense]
        if obj_sense in convention:
            convention = convention[obj_sense]

        OBJ = LT.OBJECTIVE
        if OBJ in convention:
            term_exprs.append(convention[OBJ]*sum(
                obj.expr for obj in objective_list
                ))

        EQ = LT.EQUALITY
        if EQ in convention:
            # This will result in a term of '0' if the suffix is empty,
            # which may not be what we want...
            term_exprs.append(convention[EQ]*sum(
                # IMPORTANT: We convert equalities into "canonical form"
                # by subtracting the right hand side from the body.
                # If a solver does the reverse, we must take this into
                # account in that solver's "convention."
                mult*(con.body - con.upper)
                for con, mult in suffix_map[EQ].items()
                ))

        UB = LT.PRIMAL_BOUND_UPPER
        if UB in convention:
            # Solver uses primal upper bounds in objective.
            if BD.UPPER in bound_convention:
                # TODO: Make sure "convention" and "bound_convention"
                # are compatible so this check isn't necessary.
                if bound_convention[BD.UPPER] == IC.LESS_THAN_ZERO:
                    term_exprs.append(convention[UB]*sum(
                        mult*(var - var.ub)
                        for var, mult in suffix_map[UB].items()
                        ))
                elif bound_convention[BD.UPPER] == IC.GREATER_THAN_ZERO:
                    term_exprs.append(convention[UB]*sum(
                        mult*(var.ub - var)
                        for var, mult in suffix_map[UB].items()
                        ))
                else:
                    raise RuntimeError()


        return sum(term_exprs)

    def get_gradient_lagrangian(self, convention, bound_convention=None):
        """
        Gradient of the Lagrangian with respect to primal variables,
        according to the provided convention.
        """
        model = self._model
        suffix_map = self._multiplier_suffix_map
        self._check_compatible_convention(convention)
        LT = LagrangianTerms
        BD = BoundDirection
        IC = InequalityConvention

        term_exprs = []

        objective_list = list(
                model.component_data_objects(Objective, active=True)
                )
        if len(objective_list) == 1:
            # Only infer sense from objective if exactly one objective is
            # provided. Unclear if anything other than this should be
            # supported.
            obj_sense = objective_list[0].sense
        else:
            # Not sure if anything else should be supported...
            raise RuntimeError()

        if obj_sense in bound_convention:
            bound_convention = bound_convention[obj_sense]
        if obj_sense in convention:
            convention = convention[obj_sense]

        derivs = ComponentMap([
            (var, 0.0) for var in model.component_data_objects(Var)
            if not var.fixed])

        OBJ = LT.OBJECTIVE
        if OBJ in convention:
            for obj in model.component_data_objects(Objective, active=True):
                grad = reverse_ad(obj.expr)
                for var, val in grad.items():
                    if var in derivs:
                        derivs[var] += convention[OBJ]*val

        EQ = LT.EQUALITY
        if EQ in convention:
            for con, mult in suffix_map[EQ].items():
                grad = reverse_ad(con.body-con.upper)
                for var, val in grad.items():
                    if var in derivs:
                        derivs[var] += (
                                convention[EQ]*mult*val
                                )

        UB = LT.PRIMAL_BOUND_UPPER
        if UB in convention:
            if BD.UPPER in bound_convention:
                if bound_convention[BD.UPPER] == IC.LESS_THAN_ZERO:
                    # x - xU <= 0
                    deriv_factor = 1.0
                elif bound_convention[BD.UPPER] == IC.GREATHER_THAN_ZERO:
                    # xU - x >= 0
                    deriv_factor = -1.0
                for var, mult in suffix_map[UB].items():
                    derivs[var] += (
                            convention[UB]*deriv_factor*mult
                            )

        return derivs
