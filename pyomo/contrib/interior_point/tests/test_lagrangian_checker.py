#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.contrib.interior_point.lagrangian_checker import (
        LagrangianTerms,
        LagrangianChecker,
        get_multiplier_conversion_factors,
        )


def _add_multiplier_suffixes(model):
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.dual_lb = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.dual_ub = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    LT = LagrangianTerms
    return {
            LT.EQUALITY: model.dual,
            }


class MockSolver(object):

    def solve(self, model):
        if model._model_id == 1:
            self._solve_model_1(model)
        elif model._model_id == 2:
            self._solve_model_2(model)
        else:
            raise RuntimeError()


class MockSolver1(MockSolver):
    """
    This solver uses the following convention for its Lagrangian:
        (i) All terms have a factor of +1
       (ii) Inequalities are not reformulated into slacks
      (iii) Inequality/bound multipliers are positive for minimization
            problems, negative for maximization problems.

    (i) and (iii) imply that bound and inequality multipliers are
    reformulated into "<= 0" form, regardless of original direction
    and objective sense.
    This is because "grad L == 0" must imply that we cannot improve
    the objective while remaining feasible.

    This is a convention used in many academic contexts, e.g.
    Nonlinear Programming (Biegler), chapter 4.
    """

    _LT = LagrangianTerms
    convention = {
            _LT.OBJECTIVE: 1.0,
            _LT.EQUALITY: 1.0,
            _LT.PRIMAL_BOUND_UPPER: 1.0,
            _LT.PRIMAL_BOUND_LOWER: 1.0,
            }

    def _solve_model_1(self, model):
        # Values obtained by solving by hand
        model.dual[model.eq_con] = -2.82842712
        model.v1 = 1.18920712
        model.v2 = 0.84089642

    def _solve_model_2(self, model):
        # Values obtained by solving by hand, with a priori knowledge
        # of active set. Validated with Ipopt.
        model.v1 = 2.0
        model.v2 = 0.5
        model.dual[model.eq_con] = -1.0
        model.dual_lb[model.v1] = 3.5
        model.dual_lb[model.v2] = 0.0


class MockSolver2(MockSolver):
    """
    This solver uses the following convention for its Lagrangian:
        (i) The objective has a factor of +1, multiplier terms have
            factors of -1.
       (ii) Inequalities are not reformulated into slacks
      (iii) Lower bounds/GEQ constraints have positive multipliers and
            upper bounds/LEQ constraints have negative multipliers for 
            minimization problems. Here "GEQ/LEQ constraints" are with
            respect to the constraint body. The signs are switched for
            maximization problems.

    (i) and (iii) imply that bounds and inequality constraints are
    reformulated into:
    x - x_L    >= 0
    x - x_U    <= 0
    c(x) - c_L >= 0
    c(x) - c_U <= 0

    This is the convention used by IPOPT's AMPL interface, CPLEX, and
    Gurobi.
    """
    
    _LT = LagrangianTerms
    convention = {
            _LT.OBJECTIVE: 1.0,
            _LT.EQUALITY: -1.0,
            _LT.PRIMAL_BOUND_UPPER: -1.0,
            _LT.PRIMAL_BOUND_LOWER: -1.0,
            }

    def _solve_model_1(self, model):
        model.dual[model.eq_con] = 2.82842712
        model.v1 = 1.18920712
        model.v2 = 0.84089642

    def _solve_model_2(self, model):
        # Values obtained by solving by hand, with a priori knowledge
        # of active set. Validated with Ipopt.
        model.v1 = 2.0
        model.v2 = 0.5
        model.dual[model.eq_con] = 1.0
        model.dual_lb[model.v1] = 3.5
        model.dual_lb[model.v2] = 0.0


class TestModel(unittest.TestCase):
    def assertFeasible(self, model, tol=None):
        if tol is None:
            tol = 0.0
        for con in model.component_data_objects(pyo.Constraint, active=True):
            if con.lower is None:
                # "less than" constraint
                self.assertLessEqual(
                        pyo.value(con.body),
                        pyo.value(con.upper)+tol,
                        )
            elif con.upper is None:
                # "greater than" constraint
                self.assertLessEqual(
                        pyo.value(con.lower)-tol,
                        pyo.value(con.body),
                        )
            elif pyo.value(con.lower) == pyo.value(con.upper):
                # Equality constraint
                self.assertAlmostEqual(
                        pyo.value(con.lower),
                        pyo.value(con.body),
                        delta=tol,
                        )
            else:
                # Ranged inequality
                self.assertLessEqual(
                        pyo.value(con.lower)-tol,
                        pyo.value(con.body),
                        )
                self.assertLessEqual(
                        pyo.value(con.body),
                        pyo.value(con.upper)+tol,
                        )


class TestModel1(TestModel):

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var(initialize=1.5)
        m.v2 = pyo.Var(initialize=1.5)
        m.eq_con = pyo.Constraint(expr=m.v1*m.v2 - 1 == 0)
        m.obj = pyo.Objective(expr=m.v1**2 + 2*m.v2**2, sense=pyo.minimize)
    
        # Add an "id" tag so my "solver" can hard-code the correct multipliers.
        m._model_id = 1
        return m

    def test_solver_1(self):
        m = self._make_model()
        multiplier_map = _add_multiplier_suffixes(m)

        solver = MockSolver1()
        solver.solve(m)

        self.assertFeasible(m, tol=1e-8)

        lag_check = LagrangianChecker(m, multiplier_map)

        grad_lag = lag_check.get_gradient_lagrangian(solver.convention)
        n_primals = len(grad_lag)
        
        self.assertEqual(n_primals, 2)

        self.assertStructuredAlmostEqual(
                list(grad_lag.values()),
                [0.0]*n_primals,
                delta=1e-7,
                )

    def test_solver_2(self):
        m = self._make_model()
        multiplier_map = _add_multiplier_suffixes(m)

        solver = MockSolver2()
        solver.solve(m)

        self.assertFeasible(m, tol=1e-8)

        lag_check = LagrangianChecker(m, multiplier_map)

        grad_lag = lag_check.get_gradient_lagrangian(solver.convention)
        n_primals = len(grad_lag)
        
        self.assertEqual(n_primals, 2)

        self.assertStructuredAlmostEqual(
                list(grad_lag.values()),
                [0.0]*n_primals,
                delta=1e-7,
                )

    @unittest.skipUnless(pyo.SolverFactory("ipopt").available(),
            "IPOPT is not available")
    def test_ipopt(self):
        m = self._make_model()
        multiplier_map = _add_multiplier_suffixes(m)

        solver = pyo.SolverFactory("ipopt")
        solver.solve(m)

        LT = LagrangianTerms
        ipopt_convention = {
                LT.OBJECTIVE: 1.0,
                LT.EQUALITY: -1.0,
                }
        # As this model only has equality constraints, it is only
        # necessary to specify the convention this far.

        self.assertAlmostEqual(m.dual[m.eq_con], 2.82842712, delta=1e-7)

        self.assertFeasible(m, tol=1e-8)
        lag_check = LagrangianChecker(m, multiplier_map)
        grad_lag = lag_check.get_gradient_lagrangian(ipopt_convention)
        n_primals = len(grad_lag)
        self.assertStructuredAlmostEqual(
                list(grad_lag.values()),
                [0.0]*n_primals,
                delta=1e-7,
                )

    def test_convert_multipliers_1_to_2(self):
        m = self._make_model()
        multiplier_map = _add_multiplier_suffixes(m)

        solver1 = MockSolver1()
        solver1.solve(m)

        solver2 = MockSolver2()

        self.assertFeasible(m, tol=1e-8)
        lag_check = LagrangianChecker(m, multiplier_map)
        grad_lag = lag_check.get_gradient_lagrangian(solver1.convention)
        n_primals = len(grad_lag)
        self.assertEqual(n_primals, 2)
        self.assertStructuredAlmostEqual(
                list(grad_lag.values()),
                [0.0]*n_primals,
                delta=1e-7,
                )

        conv_factors = get_multiplier_conversion_factors(
                solver1.convention,
                None,
                solver2.convention,
                None,
                )

        for term, factor in conv_factors.items():
            suffix = multiplier_map[term]
            for comp in suffix:
                suffix[comp] *= factor

        grad_lag = lag_check.get_gradient_lagrangian(solver2.convention)
        n_primals = len(grad_lag)
        self.assertEqual(n_primals, 2)
        self.assertStructuredAlmostEqual(
                list(grad_lag.values()),
                [0.0]*n_primals,
                delta=1e-7,
                )


class TestModel2(TestModel):

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var(initialize=1.5, bounds=(2.0, None))
        m.v2 = pyo.Var(initialize=1.5, bounds=(0.0, None))
        m.eq_con = pyo.Constraint(expr=m.v1*m.v2 - 1 == 0)
        m.obj = pyo.Objective(expr=m.v1**2 + 2*m.v2**2, sense=pyo.minimize)
    
        # Add an "id" tag so my "solver" can hard-code the correct multipliers.
        m._model_id = 2
        return m

    def test_solver_1(self):
        m = self._make_model()
        multiplier_map = _add_multiplier_suffixes(m)

        solver = MockSolver1()
        solver.solve(m)

        self.assertFeasible(m, tol=1e-8)

        lag_check = LagrangianChecker(m, multiplier_map)

        # TODO: implement bound terms in LagrangianChecker
        grad_lag = lag_check.get_gradient_lagrangian(solver.convention)
        n_primals = len(grad_lag)
        
        self.assertEqual(n_primals, 2)

        self.assertStructuredAlmostEqual(
                list(grad_lag.values()),
                [0.0]*n_primals,
                delta=1e-7,
                )


if __name__ == "__main__":
    unittest.main()
