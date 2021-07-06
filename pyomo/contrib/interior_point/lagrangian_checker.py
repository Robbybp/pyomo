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


class Conventions(enum.Enum):
    """ The conventions we must be aware of in order to construct a
    Lagrangian function from a Pyomo model.
    """
    TERM_FACTORS = 0
    INEQUALITY_DIRECTION = 1
    SLACK_CONVENTION = 2


class LagrangianTerms(enum.Enum):
    # The objective term in the Lagrangian may have an arbitrary nonzero
    # factor applied to it. If a solver treats maximization problems as
    # "minimization problems with a negative objective," that should be
    # reflected in this term's factor for maximization problems.
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
    # TODO: Why don't we use pyo.minimize and pyo.maximize again?
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


def _check_compatible_conventions(factors, inequalities):
    LT = LagrangianTerms
    equality_terms = {
            LT.EQUALITY,
            LT.INEQUALITY_LOWER_PLUS_SLACK,
            LT.INEQUALITY_UPPER_PLUS_SLACK,
            LT.INEQUALITY_LOWER_MINUS_SLACK,
            LT.INEQUALITY_UPPER_MINUS_SLACK,
            }
    for term in factors:
        if term != LT.OBJECTIVE and term not in equality_terms:
            # Make sure every non-objective, non-equality term
            # has a convention to reformulate it as ">= 0" or
            # "<= 0"
            if term not in inequalities:
                raise RuntimeError(
                    "Was not provided an inequality direction\n"
                    "convention for term %s." % term
                    )


def get_multiplier_conversion_factors(source_convention, target_convention):
        #source_factors,
        #source_inequality_signs,
        #target_factors,
        #target_inequality_signs,
        #):
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

    We assume that the input data structures do not contain objective senses.
    If they do, the conversion may be ambiguous. (What conversion is necessary
    may depend on whether we are maximizing or minimizing.)
    """
    Conv = Conventions
    # Factors for each term: required
    source_factors = source_convention[Conv.TERM_FACTORS]
    target_factors = target_convention[Conv.TERM_FACTORS]

    # Direction for each type of inequality: optional. Not all solvers
    # support inequalities.
    source_ineq_form = source_convention.get(Conv.INEQUALITY_DIRECTION, {})
    target_ineq_form = target_convention.get(Conv.INEQUALITY_DIRECTION, {})

    # TODO: Not sure exactly what will go here. What are the different
    # conventions for slack variables?
    source_slack = source_convention.get(Conv.SLACK_CONVENTION, None)
    target_slack = target_convention.get(Conv.SLACK_CONVENTION, None)

    _check_compatible_conventions(source_factors, source_ineq_form)
    _check_compatible_conventions(target_factors, target_ineq_form)
    
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

    LB = LT.PRIMAL_BOUND_LOWER
    if LB in source_factors:
        inequality_factor = 1.0 if (source_ineq_form[LB] ==
                target_ineq_form[LB]) else -1.0
        conversion_factors[LB] = (
                objective_factor * inequality_factor *
                source_factors[LB] / target_factors[LB]
                )
    
    UB = LT.PRIMAL_BOUND_UPPER
    if UB in source_factors:
        inequality_factor = 1.0 if (source_ineq_form[UB] ==
                target_ineq_form[UB]) else -1.0
        conversion_factors[UB] = (
                objective_factor * inequality_factor *
                source_factors[UB] / target_factors[UB]
                )

    return conversion_factors


class LagrangianChecker(object):

    def __init__(self, model, multiplier_suffix_map):
        """
        To construct the gradient of the Lagrangian, we need model
        variables/bounds/constraints, and the corresponding multipliers.

        Different solvers may use different suffixes to store the
        multipliers, so we require a map from LagrangianTerms enum items
        to the suffix used for that term's multipliers. E.g. for Ipopt:

        >>> multiplier_suffix_map = {
        >>>     LagrangianTerms.EQUALITY: model.dual,
        >>>     LagrangianTerms.PRIMAL_BOUND_LOWER: model.ipopt_zL_out,
        >>>     LagrangianTerms.PRIMAL_BOUND_UPPER: model.ipopt_zU_out,
        >>>     }

        """
        self._model = model
        self._multiplier_suffix_map = multiplier_suffix_map

    def _check_compatible_convention(self, convention):
        """
        A "convention" contains all the necessary information to construct
        the Lagrangian given variables, constraints, bounds, and
        multipliers.

        It is possible/common for the "convention" to change depending on
        whether we are maximizing or minimizing. It is also possible for
        this "convention" to not change, but for the signs of the
        multipliers to change instead.

        If the convention contains terms corresponding to inequalities,
        it must also contain the required information necessary to
        construct the corresponding terms in the Lagrangian. This is
        (a) whether slack variables are used (for inequality constraints)
        and (b) the inequality direction (greater or less than zero).
        This method checks whether such information is provided for
        terms corresponding to inequalities.

        """
        suffix_map = self._multiplier_suffix_map

        OS = ObjectiveSense
        convention_dict = {}
        bound_convention_dict = {}

        # If no sense is specified, assume the provided conventions
        # hold for any objective sense.
        if all(sense not in convention for sense in OS):
            conventions_to_check = [convention]
            supported_senses = set(OS)
        else:
            conventions_to_check = [convention[sense] for sense in OS
                    if sense in convention]
            supported_senses = set(s for s in OS if s in convention)

        Conv = Conventions
        LT = LagrangianTerms
        for conv in conventions_to_check:
            factors = conv[Conv.TERM_FACTORS]
            inequalities = conv.get(Conv.INEQUALITY_DIRECTION, {})
            # TODO: Slacks

            _check_compatible_conventions(factors, inequalities)

            for term in factors:
                if term != LT.OBJECTIVE:
                    if term not in suffix_map:
                        raise RuntimeError(
                            "Was not provided a suffix for Lagrangian term %s."
                            % term
                            )

        return supported_senses

    def get_gradient_lagrangian(self, convention):
        """
        Gradient of the Lagrangian with respect to primal variables,
        according to the provided convention.
        """
        Conv = Conventions

        model = self._model
        suffix_map = self._multiplier_suffix_map
        supported_senses = self._check_compatible_convention(convention)
        LT = LagrangianTerms
        IC = InequalityConvention
        OS = ObjectiveSense

        term_exprs = []

        objective_list = list(
                model.component_data_objects(Objective, active=True)
                )
        if len(objective_list) == 1:
            # Only infer sense from objective if exactly one objective is
            # provided. Unclear if anything other than this should be
            # supported.
            obj_sense = objective_list[0].sense
            obj_sense = OS.MAXIMIZE if obj_sense == maximize else OS.MINIMIZE
        else:
            # Not sure if anything else should be supported...
            raise RuntimeError()

        if obj_sense in supported_senses:
            if obj_sense in convention:
                convention = convention[obj_sense]
        else:
            # Our objective has a sense that is not supported by the
            # convention provided.
            raise RuntimeError()

        factors = convention[Conv.TERM_FACTORS]
        inequalities = convention.get(Conv.INEQUALITY_DIRECTION, {})
        # TODO: Slacks

        derivs = ComponentMap([
            (var, 0.0) for var in model.component_data_objects(Var)
            if not var.fixed])

        OBJ = LT.OBJECTIVE
        if OBJ in factors:
            for obj in model.component_data_objects(Objective, active=True):
                grad = reverse_ad(obj.expr)
                for var, val in grad.items():
                    if var in derivs:
                        derivs[var] += factors[OBJ]*val

        EQ = LT.EQUALITY
        if EQ in factors:
            for con, mult in suffix_map[EQ].items():
                grad = reverse_ad(con.body-con.upper)
                for var, val in grad.items():
                    if var in derivs:
                        # Need to check because reverse_ad may compute
                        # derivatives with respect to fixed vars.
                        derivs[var] += (
                                factors[EQ]*mult*val
                                )

        UB = LT.PRIMAL_BOUND_UPPER
        if UB in factors:
            # May assume that UB in bound_convention as well
            if inequalities[UB] == IC.LESS_THAN_ZERO:
                # x - xU <= 0
                deriv_factor = 1.0
            elif inequalities[UB] == IC.GREATER_THAN_ZERO:
                # xU - x >= 0
                deriv_factor = -1.0
            for var, mult in suffix_map[UB].items():
                derivs[var] += (
                        factors[UB]*deriv_factor*mult
                        )

        LB = LT.PRIMAL_BOUND_LOWER
        if LB in factors:
            # May assume that LB in bound_convention as well
            if inequalities[LB] == IC.LESS_THAN_ZERO:
                # xL - x <= 0
                deriv_factor = -1.0
            elif inequalities[LB] == IC.GREATER_THAN_ZERO:
                # x - xL >= 0
                deriv_factor = 1.0
            for var, mult in suffix_map[LB].items():
                derivs[var] += (
                        factors[LB]*deriv_factor*mult
                        )

        return derivs
