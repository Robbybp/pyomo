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

    # The following terms are due to constraints. Pyomo stores constraints
    # in the "cL <= c(x) <= cU" format, which appear differently in the
    # Lagrangian depending on whether they are equalities (cL == cU)
    # or inequalities (cL < cU).

    # The following terms are due to equalities. We assume these terms are
    # "formulated" as follows, i.e. the signs of the constraint bodies are
    # not altered. If they are, this should be reflected in the solver's
    # "Lagrangian factors" data structure.
    EQUALITY = 1                       #          c(x) == 0
    INEQUALITY_LOWER_PLUS_SLACK = 2    # c(x) + s - cL == 0
    INEQUALITY_UPPER_PLUS_SLACK = 3    # c(x) + s - cU == 0
    INEQUALITY_LOWER_MINUS_SLACK = 4   # c(x) - s - cL == 0
    INEQUALITY_UPPER_MINUS_SLACK = 5   # c(x) - s - cU == 0

    # The following terms are due to inequalities (including bounds). Each
    # of these may be reformulated into a "<= 0" or ">= 0" constraint.
    # This must be specified by the solver's "bound convention" data
    # structure.
    INEQUALITY_LOWER = 6   #   cL <= c(x)
    INEQUALITY_UPPER = 7   # c(x) <= cU
    PRIMAL_BOUND_LOWER = 8 #   xL <= x
    PRIMAL_BOUND_UPPER = 9 #    x <= xU
    SLACK_BOUND_UPPER = 10 #    0 <= s
    SLACK_BOUND_LOWER = 11 #    s <= cU - cL


class ObjectiveSense(enum.Enum):
    MINIMIZE = 0
    MAXIMIZE = 1


class InequalityConvention(enum.Enum):
    LESS_THAN_ZERO = 0
    GREATER_THAN_ZERO = 1


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

    def _check_compatible_convention(self, convention, bound_convention=None):
        suffix_map = self._multiplier_suffix_map

        if bound_convention is None:
            # A bound convention is not required if the model is only
            # equality constrained
            bound_convention = {}

        OS = ObjectiveSense
        convention_dict = {}
        bound_convention_dict = {}

        # If no sense is specified, assume the provided conventions
        # hold for any objective sense.
        if all(sense not in convention_dict for sense in OS):
            for sense in OS:
                convention_dict[sense] = convention
        if all(sense not in bound_convention_dict for sense in OS):
            for sense in OS:
                bound_convention_dict[sense] = bound_convention

        supported_senses = [s for s in convention_dict
                if s in bound_convention_dict]
        conventions_to_check = [(convention_dict[s], bound_convention_dict[s])
                for s in supported_senses]

        LT = LagrangianTerms
        for conv, bound_conv in conventions_to_check:
            for term in conv:
                if term != LT.OBJECTIVE:
                    # Make sure every non-objective term the solver expects
                    # has a suffix for the multiplier values.
                    #
                    # Extra suffixes are fine.
                    if term not in suffix_map:
                        raise RuntimeError(
                            "Was not provided a suffix for Lagrangian term %s."
                            % term
                            )

                    if (term != LT.EQUALITY
                        and term != LT.INEQUALITY_LOWER_PLUS_SLACK
                        and term != LT.INEQUALITY_UPPER_PLUS_SLACK
                        and term != LT.INEQUALITY_LOWER_MINUS_SLACK
                        and term != LT.INEQUALITY_UPPER_MINUS_SLACK):
                        # Make sure every non-objective, non-equality term
                        # has a convention to reformulate it as ">= 0" or
                        # "<= 0"
                        if term not in bound_conv:
                            raise RuntimeError(
                                "Was not provided an inequality direction\n"
                                "convention for term %s." % term
                                )

    def get_lagrangian(self, convention, bound_convention=None):
        model = self._model
        suffix_map = self._multiplier_suffix_map
        self._check_compatible_convention(convention,
                bound_convention=bound_convention)
        LT = LagrangianTerms
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
        self._check_compatible_convention(convention,
                bound_convention=bound_convention)
        LT = LagrangianTerms
        IC = InequalityConvention

        if bound_convention is None:
            bound_convention = {}

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
                        # Need to check because reverse_ad may compute
                        # derivatives with respect to fixed vars.
                        derivs[var] += (
                                convention[EQ]*mult*val
                                )

        UB = LT.PRIMAL_BOUND_UPPER
        if UB in convention:
            # May assume that UB in bound_convention as well
            if bound_convention[UB] == IC.LESS_THAN_ZERO:
                # x - xU <= 0
                deriv_factor = 1.0
            elif bound_convention[UB] == IC.GREATER_THAN_ZERO:
                # xU - x >= 0
                deriv_factor = -1.0
            for var, mult in suffix_map[UB].items():
                derivs[var] += (
                        convention[UB]*deriv_factor*mult
                        )

        LB = LT.PRIMAL_BOUND_LOWER
        if LB in convention:
            # May assume that LB in bound_convention as well
            if bound_convention[LB] == IC.LESS_THAN_ZERO:
                # xL - x <= 0
                deriv_factor = -1.0
            elif bound_convention[LB] == IC.GREATER_THAN_ZERO:
                # x - xL >= 0
                deriv_factor = 1.0
            for var, mult in suffix_map[LB].items():
                derivs[var] += (
                        convention[LB]*deriv_factor*mult
                        )

        return derivs
