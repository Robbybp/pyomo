#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
from pyomo.contrib.interior_point.lagrangian_checker import (
        LagrangianTerms,
        LagrangianChecker,
        get_conversion_factors,
        )

def _add_multiplier_suffixes(m):
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.dual_lb = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.dual_ub = pyo.Suffix(direction=pyo.Suffix.IMPORT)


def make_model_1():
    m = pyo.ConcreteModel()
    m.v1 = pyo.Var(initialize=1.5)
    m.v2 = pyo.Var(initialize=1.5)
    m.eq_con = pyo.Constraint(expr=m.v1*m.v2 - 1 == 0)
    m.obj = pyo.Objective(expr=m.v1**2 + 2*m.v2**2, sense=pyo.minimize)

    # Add an "id" tag so my "solver" can hard-code the correct multipliers.
    m._model_id = 1
    return m


class MockSolver1(object):
    """
    This solver uses the following convention for its Lagrangian:
        (i) All terms have a factor of +1
       (ii) Inequalities are not reformulated into slacks
      (iii) Inequality/bound multipliers are positive for minimization
            problems, negative for maximization problems.

    (i) and (iii) imply that bound and inequality multipliers are
    reformulated into ">= 0" form, regardless of original direction
    or objective sense.
    """

    def solve(self, model):
        if model._model_id == 1:
            self._solve_model_1(model)
        else:
            raise RuntimeError()

    def _solve_model_1(self, model):
        pass
