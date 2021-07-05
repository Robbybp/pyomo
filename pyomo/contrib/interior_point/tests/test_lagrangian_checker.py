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
        Conventions,
        LagrangianTerms,
        InequalityConvention,
        LagrangianChecker,
        ObjectiveSense,
        get_multiplier_conversion_factors,
        )


def _add_multiplier_suffixes(model):
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.dual_lb = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.dual_ub = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    LT = LagrangianTerms
    return {
            LT.EQUALITY: model.dual,
            LT.PRIMAL_BOUND_LOWER: model.dual_lb,
            LT.PRIMAL_BOUND_UPPER: model.dual_ub,
            }


class MockSolver(object):
    """
    The purpose of this class is just to not repeat the following
    solve method for all the mock solvers in this module.
    """
    def solve(self, model):
        if model._model_id == 1:
            self._solve_model_1(model)
        elif model._model_id == 2:
            self._solve_model_2(model)
        elif model._model_id == 3:
            self._solve_model_3(model)
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
    _IC = InequalityConvention
    _C = Conventions
    convention = {
            _C.TERM_FACTORS: {
                _LT.OBJECTIVE: 1.0,
                _LT.EQUALITY: 1.0,
                _LT.PRIMAL_BOUND_UPPER: 1.0,
                _LT.PRIMAL_BOUND_LOWER: 1.0,
                },
            _C.INEQUALITY_DIRECTION: {
                _LT.PRIMAL_BOUND_UPPER: _IC.LESS_THAN_ZERO,
                _LT.PRIMAL_BOUND_LOWER: _IC.LESS_THAN_ZERO,
                },
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

    def _solve_model_3(self, model):
        # Values obtained by solving by hand.
        model.v1 = -2.0
        model.v2 = 1.5
        model.v3 = -2.0/3.0
        model.dual[model.eq_con] = 4.0/9.0
        model.dual_ub[model.v1] = -32.0/9.0
        model.dual_lb[model.v2] = -65.0/27.0


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
    _IC = InequalityConvention
    _C = Conventions
    convention = {
            _C.TERM_FACTORS: {
                _LT.OBJECTIVE: 1.0,
                _LT.EQUALITY: -1.0,
                _LT.PRIMAL_BOUND_UPPER: -1.0,
                _LT.PRIMAL_BOUND_LOWER: -1.0,
                },
            _C.INEQUALITY_DIRECTION: {
                _LT.PRIMAL_BOUND_LOWER: _IC.GREATER_THAN_ZERO,
                _LT.PRIMAL_BOUND_UPPER: _IC.LESS_THAN_ZERO,
                },
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

    def _solve_model_3(self, model):
        # Values obtained by solving by hand.
        model.v1 = -2.0
        model.v2 = 1.5
        model.v3 = -2.0/3.0
        model.dual[model.eq_con] = -4.0/9.0
        model.dual_ub[model.v1] = 32.0/9.0
        model.dual_lb[model.v2] = -65.0/27.0


class MockSolver3(MockSolver):
    """
    A somewhat pathological solver for testing purposes. This solver
    changes the form of its Lagrangian depending on whether it is
    solving maximization or minimization problems. Furthermore,
    it uses factors for its Lagrangian terms that have magnitudes
    different than 1.

    This solver uses the following convention for its Lagrangian:
        (i) The objective has a factor of +1, multiplier terms have
            factors of +10.
       (ii) Inequalities are not reformulated into slacks
      (iii) Bound and inequality multipliers are always positive.
            
    (i) and (iii) imply that bounds and inequality constraints are
    reformulated into ">= 0" form for maximization problems and
    "<= 0" form for minimization problems.
    """

    _LT = LagrangianTerms
    _IC = InequalityConvention
    _C = Conventions
    _OS = ObjectiveSense
    convention = {
            _OS.MAXIMIZE: {
                _C.TERM_FACTORS: {
                    _LT.OBJECTIVE: 1.0,
                    _LT.EQUALITY: 10.0,
                    _LT.PRIMAL_BOUND_UPPER: 10.0,
                    _LT.PRIMAL_BOUND_LOWER: 10.0,
                    },
                _C.INEQUALITY_DIRECTION: {
                    _LT.PRIMAL_BOUND_LOWER: _IC.GREATER_THAN_ZERO,
                    _LT.PRIMAL_BOUND_UPPER: _IC.GREATER_THAN_ZERO,
                    },
                },
            _OS.MINIMIZE: {
                _C.TERM_FACTORS: {
                    _LT.OBJECTIVE: 1.0,
                    _LT.EQUALITY: 10.0,
                    _LT.PRIMAL_BOUND_UPPER: 10.0,
                    _LT.PRIMAL_BOUND_LOWER: 10.0,
                    },
                _C.INEQUALITY_DIRECTION: {
                    _LT.PRIMAL_BOUND_LOWER: _IC.LESS_THAN_ZERO,
                    _LT.PRIMAL_BOUND_UPPER: _IC.LESS_THAN_ZERO,
                    },
                },
            }


class TestModel(unittest.TestCase):
    """
    This class adds the assertFeasible method and some utilities
    to test a given solver with a given model.
    """
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

    def _test_solver(self, solver, model_info=None):
        """
        model_info is a tuple containing the model and the
        "multiplier map" dictionary.
        """
        if model_info is None:
            # Make model with suffixes
            m = self._make_model()
            multiplier_map = _add_multiplier_suffixes(m)
        else:
            # If we want to make a model with custom suffixes
            # for a specific (non-dummy, presumably) solver.
            m, multiplier_map = model_info

        # Solve and assertFeasible
        solver.solve(m)
        self.assertFeasible(m, tol=1e-7)

        # Create gradient-of-Lagrangian data structure
        lag_check = LagrangianChecker(m, multiplier_map)
        grad_lag = lag_check.get_gradient_lagrangian(solver.convention)
        n_primals = len(grad_lag)
        
        # Assert gradient-of-Lagrangian is what we expect (zero)
        n_vars = len([v for v in m.component_data_objects(pyo.Var)
            if not v.fixed])
        self.assertEqual(n_primals, n_vars)
        self.assertStructuredAlmostEqual(
                list(grad_lag.values()),
                [0.0]*n_primals,
                delta=1e-7,
                )

        return lag_check

    def _test_convert_multipliers(self, solver1, solver2, model_info=None):
        # Test that the first solver can solve the model, and that
        # its convention and the provided multipliers lead to a consistent
        # Lagrangian gradient.
        lag_check = self._test_solver(solver1, model_info=model_info)
        m = lag_check._model
        multiplier_map = lag_check._multiplier_suffix_map

        # Get conversion factors between the two solvers' conventions
        conv_factors = get_multiplier_conversion_factors(
                solver1.convention,
                solver2.convention,
                )

        # Convert multipliers
        for term, factor in conv_factors.items():
            suffix = multiplier_map[term]
            for comp in suffix:
                suffix[comp] *= factor

        n_var = len([v for v in m.component_data_objects(pyo.Var)
            if not v.fixed])

        # Check that the gradient of the Lagrangian, now with solver2's
        # convention, is zero.
        grad_lag = lag_check.get_gradient_lagrangian(solver2.convention)
        n_primals = len(grad_lag)
        self.assertEqual(n_primals, n_var)
        self.assertStructuredAlmostEqual(
                list(grad_lag.values()),
                [0.0]*n_primals,
                delta=1e-7,
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
        solver = MockSolver1()
        self._test_solver(solver)

    def test_solver_2(self):
        solver = MockSolver2()
        self._test_solver(solver)

    @unittest.skipUnless(pyo.SolverFactory("ipopt").available(),
            "IPOPT is not available")
    def test_ipopt(self):
        solver = pyo.SolverFactory("ipopt")
        LT = LagrangianTerms
        Conv = Conventions
        solver.convention = {
                Conv.TERM_FACTORS: {
                    LT.OBJECTIVE: 1.0,
                    LT.EQUALITY: -1.0,
                    }
                }
        #solver.bound_convention = {}
        self._test_solver(solver)

    def test_convert_multipliers_1_to_2(self):
        solver1 = MockSolver1()
        solver2 = MockSolver2()
        self._test_convert_multipliers(solver1, solver2)


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
        solver = MockSolver1()
        self._test_solver(solver)

    def test_solver_2(self):
        solver = MockSolver2()
        self._test_solver(solver)

    @unittest.skipUnless(pyo.SolverFactory("ipopt").available(),
            "IPOPT is not available")
    def test_ipopt(self):
        solver = pyo.SolverFactory("ipopt")
        LT = LagrangianTerms
        IC = InequalityConvention
        Conv = Conventions
        solver.convention = {
                Conv.TERM_FACTORS: {
                    LT.OBJECTIVE: 1.0,
                    LT.EQUALITY: -1.0,
                    LT.PRIMAL_BOUND_LOWER: -1.0,
                    LT.PRIMAL_BOUND_UPPER: -1.0,
                    },
                Conv.INEQUALITY_DIRECTION: {
                    LT.PRIMAL_BOUND_LOWER: IC.GREATER_THAN_ZERO,
                    LT.PRIMAL_BOUND_UPPER: IC.LESS_THAN_ZERO,
                    },
                }
        m = self._make_model()
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        multiplier_map = {
                LT.EQUALITY: m.dual,
                LT.PRIMAL_BOUND_LOWER: m.ipopt_zL_out,
                LT.PRIMAL_BOUND_UPPER: m.ipopt_zU_out,
                }
        self._test_solver(solver, model_info=(m, multiplier_map))

    def test_convert_multipliers_1_to_2(self):
        solver1 = MockSolver1()
        solver2 = MockSolver2()
        self._test_convert_multipliers(solver1, solver2)


class TestModel3(TestModel):
    """
    This model maximizes its objective function and has an active
    upper bound at the solution.
    """

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var(initialize=-2.5, bounds=(None, -2.0))
        m.v2 = pyo.Var(initialize=2.5, bounds=(1.5, None))
        m.v3 = pyo.Var(initialize=-2.5, bounds=(None, 0.0))

        m.eq_con = pyo.Constraint(expr=m.v1*m.v2*m.v3 - 2.0 == 0)
        m.obj = pyo.Objective(expr=-m.v1**2 - m.v2**2 - m.v3**2,
                sense=pyo.maximize)
    
        # Add an "id" tag so my "solver" can hard-code the correct multipliers.
        m._model_id = 3
        return m

    def test_solver_1(self):
        solver = MockSolver1()
        self._test_solver(solver)

    def test_solver_2(self):
        solver = MockSolver2()
        self._test_solver(solver)

    def test_convert_multipliers_1_to_2(self):
        solver1 = MockSolver1()
        solver2 = MockSolver2()
        self._test_convert_multipliers(solver1, solver2)

    @unittest.skipUnless(pyo.SolverFactory("ipopt").available(),
            "IPOPT is not available")
    def test_ipopt(self):
        solver = pyo.SolverFactory("ipopt")
        LT = LagrangianTerms
        IC = InequalityConvention
        Conv = Conventions
        solver.convention = {
                Conv.TERM_FACTORS: {
                    LT.OBJECTIVE: 1.0,
                    LT.EQUALITY: -1.0,
                    LT.PRIMAL_BOUND_LOWER: -1.0,
                    LT.PRIMAL_BOUND_UPPER: -1.0,
                    },
                Conv.INEQUALITY_DIRECTION: {
                    LT.PRIMAL_BOUND_LOWER: IC.GREATER_THAN_ZERO,
                    LT.PRIMAL_BOUND_UPPER: IC.LESS_THAN_ZERO,
                    },
                }
        m = self._make_model()
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        multiplier_map = {
                LT.EQUALITY: m.dual,
                LT.PRIMAL_BOUND_LOWER: m.ipopt_zL_out,
                LT.PRIMAL_BOUND_UPPER: m.ipopt_zU_out,
                }
        self._test_solver(solver, model_info=(m, multiplier_map))


if __name__ == "__main__":
    unittest.main()
