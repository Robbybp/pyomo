#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy, scipy_available
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.common.dependencies.scipy import sparse as sps

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run cyipopt tests")

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    cyipopt_available,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxModel,
    ExternalGreyBoxBlock,
    ScalarExternalGreyBoxBlock,
    IndexedExternalGreyBoxBlock,
)
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
    PressureDropTwoOutputsWithHessian,
)
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import (
    PyomoNLP,
)
from pyomo.contrib.pynumero.interfaces.functions import (
    VectorValuedExternalFunction,
    FunctionComposition,
    FunctionFromNLP,
    FunctionStack,
    FunctionCombination,
    IdentityFunction,
    NLPFromFunction,
)
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
    ProjectedNLP,
)
from pyomo.contrib.pynumero.interfaces.abstract_nlps import (
    FixedVarNLP,
)
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    CyIpoptNLP,
    CyIpoptSolver,
)

if not pyo.SolverFactory("ipopt").available():
    raise unittest.SkipTest(
        "Need IPOPT to run ExternalPyomoModel tests"
        )


class SimpleFunction1(VectorValuedExternalFunction):
    """
    Maps x to (5*x - 1, x**2 + 50)
    """

    def __init__(self):
        self.inputs = np.zeros(1)
        self.outputs = np.zeros(2)
        self.jacobian_outputs = np.zeros((2, 1))
        self.hessian_outputs = np.zeros((2, 1, 1))

    def n_inputs(self):
        return len(self.inputs)

    def n_outputs(self):
        return len(self.outputs)

    def set_input_values(self, input_values):
        self.inputs = input_values
        x = self.inputs[0]
        self.outputs[0] = 5.0*x - 1.0
        self.outputs[1] = x**2 + 50.0
        self.jacobian_outputs[0, 0] = 5.0
        self.jacobian_outputs[1, 0] = 2.0*x
        h000 = 0.0
        h100 = 2.0
        self.hessian_outputs[0, 0, 0] = h000
        self.hessian_outputs[1, 0, 0] = h100

    def evaluate_outputs(self):
        return self.outputs

    def evaluate_jacobian_outputs(self):
        return sps.coo_matrix(self.jacobian_outputs)

    def evaluate_hessian_outputs(self):
        return [sps.coo_matrix(self.hessian_outputs[i, :, :])
            for i in range(self.n_outputs())]


class SimpleFunction2(VectorValuedExternalFunction):
    """
    My vision for a vector-valued external function is an object
    with no notion of equality constraints, and no notion of names.
    Ideally, it performs all calculations in compiled code.
    """

    def __init__(self):
        self.inputs = np.zeros(2)
        self.outputs = np.zeros(1)
        self.jacobian_outputs = np.zeros((1, 2))
        self.hessian_outputs = np.zeros((1, 2, 2))

    def n_inputs(self):
        return len(self.inputs)

    def set_input_values(self, input_values):
        self.inputs = input_values
        self.outputs[0] = np.sqrt(self.inputs[0]**2 + self.inputs[1]**2)
        self.jacobian_outputs[0, 0] = self.inputs[0]/self.outputs[0]
        self.jacobian_outputs[0, 1] = self.inputs[1]/self.outputs[0]
        h000 = (self.outputs[0]**2 - self.inputs[0]**2)/self.outputs[0]**3
        h001 = - self.inputs[0]*self.inputs[1]/self.outputs[0]**3
        h011 = (self.outputs[0]**2 - self.inputs[1]**2)/self.outputs[0]**3
        self.hessian_outputs[0, 0, 0] = h000
        self.hessian_outputs[0, 1, 0] = h001
        self.hessian_outputs[0, 0, 1] = h001
        self.hessian_outputs[0, 1, 1] = h011

    def n_outputs(self):
        return len(self.outputs)

    def evaluate_outputs(self):
        return self.outputs

    def evaluate_jacobian_outputs(self):
        return sps.coo_matrix(self.jacobian_outputs)

    def evaluate_hessian_outputs(self):
        # TODO: What data format should I use for sparse tensors.
        # Want:
        # - fast access along rank 1 (to get an individual output's hessian matrix)
        # - Fast multiplication and transpose for a slice across rank 1
        # - Identification of a slice across ranks 2 and 3. This is what
        #   an array of coo matrices doesn't handle well...
        # Current choice is a list of coo matrices.
        return [sps.coo_matrix(self.hessian_outputs[i, :, :])
            for i in range(self.n_outputs())]


"""
The following four functions define models used to test
the embedding of a scalar-valued function in an NLP.
"""


def make_model1_xu():
    """
    Model in the space of x and u.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.5)
    m.u = pyo.Var(initialize=2.5)
    m.eq_con = pyo.Constraint(expr=m.x*m.u - 1.0 == 0)
    m.obj = pyo.Objective(expr=m.x**2 + 2*m.u**2)
    return m


def make_model1_yzu():
    """
    Suppose x is a function of y and z.
    x = sqrt(y**2 + z**2)
    This model is in the space of u, y, and z
    """
    m = pyo.ConcreteModel()
    m.y = pyo.Var(initialize=1.5)
    m.z = pyo.Var(initialize=1.5)
    m.u = pyo.Var(initialize=2.5)
    m.eq_con = pyo.Constraint(expr=pyo.sqrt(m.y**2 + m.z**2)*m.u - 1.0 == 0)
    m.obj = pyo.Objective(expr=m.y**2 + m.z**2 + 2*m.u**2)
    return m


def make_model2_xyzu():
    """
    Suppose y and z are already used within our model.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.5)
    m.y = pyo.Var(initialize=1.5)
    m.z = pyo.Var(initialize=1.5)
    m.u = pyo.Var(initialize=2.5)
    m.eq_con1 = pyo.Constraint(expr=m.x*m.u - 1.0 == 0)
    m.eq_con2 = pyo.Constraint(expr=m.y*m.z - 2.0 == 0)
    m.obj = pyo.Objective(expr=m.x**2 + 2*m.u**2)
    return m


def make_model2_yzu():
    m = pyo.ConcreteModel()
    m.y = pyo.Var(initialize=1.5)
    m.z = pyo.Var(initialize=1.5)
    m.u = pyo.Var(initialize=2.5)
    m.eq_con1 = pyo.Constraint(expr=pyo.sqrt(m.y**2 + m.z**2)*m.u - 1.0 == 0)
    m.eq_con2 = pyo.Constraint(expr=m.y*m.z - 2.0 == 0)
    m.obj = pyo.Objective(expr=m.y**2 + m.z**2 + 2*m.u**2)
    return m


class TestSimpleFunctionComposition(unittest.TestCase):

    def test_outputs(self):
        f = SimpleFunction1()
        g = SimpleFunction2()
        fog = FunctionComposition(f, g)

        inputs = np.array([1.0, 2.0])
        fog.set_input_values(inputs)
        outputs = fog.evaluate_outputs()

        np.testing.assert_allclose(
                outputs,
                [np.sqrt(5.0)*5.0 - 1.0, 55.0],
                )

    def test_jacobian(self):
        f = SimpleFunction1()
        g = SimpleFunction2()
        fog = FunctionComposition(f, g)

        y, z = 1.0, 2.0
        inputs = np.array([y, z])
        fog.set_input_values(inputs)
        jacobian = fog.evaluate_jacobian_outputs()
        
        jac_pred = [
                [5*y/np.sqrt(y**2 + z**2), 5*z/np.sqrt(y**2 + z**2)],
                [2*y, 2*z],
                ]
        np.testing.assert_allclose(jacobian.toarray(), jac_pred)

    def test_hessian(self):
        f = SimpleFunction1()
        g = SimpleFunction2()
        fog = FunctionComposition(f, g)

        y, z = 1.0, 2.0
        inputs = np.array([y, z])
        fog.set_input_values(inputs)
        hessian = fog.evaluate_hessian_outputs()

        denom = np.sqrt(y**2+z**2)**3
        hess_pred = [
                [[5*z**2/denom, -5*y*z/denom],
                [-5*y*z/denom, 5*y**2/denom]],
                [[2.0, 0.0],
                [0.0, 2.0]],
                ]
        for pred, act in zip(hess_pred, hessian):
            np.testing.assert_allclose(pred, act.toarray())


class TestFunctionFromNLP(unittest.TestCase):

    def test_outputs(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        fcn = FunctionFromNLP(nlp)

        self.assertEqual(fcn.n_inputs(), 2)
        self.assertEqual(fcn.n_outputs(), 2)

        x, u = 5.0, 4.0
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        input_values = np.zeros(fcn.n_inputs())
        input_values[x_idx] = x
        input_values[u_idx] = u
        fcn.set_input_values(input_values)

        outputs = fcn.evaluate_outputs()

        # Know that the objective comes before the (single) constraint,
        # so don't have to get any indices here.
        pred_outputs = [57., 19.]
        np.testing.assert_allclose(outputs, pred_outputs)

    def test_jacobian(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        fcn = FunctionFromNLP(nlp)

        x, u = 5.0, 4.0
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        input_values = np.zeros(2)
        input_values[x_idx] = x
        input_values[u_idx] = u
        fcn.set_input_values(input_values)

        jacobian = fcn.evaluate_jacobian_outputs()
        pred_jac = np.zeros((2,2))
        pred_jac[0, x_idx] = 2*x
        pred_jac[0, u_idx] = 4*u
        pred_jac[1, x_idx] = u
        pred_jac[1, u_idx] = x
        np.testing.assert_allclose(jacobian.toarray(), pred_jac)

    def test_hessian(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        fcn = FunctionFromNLP(nlp)

        x, u = 5.0, 4.0
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        input_values = np.zeros(2)
        input_values[x_idx] = x
        input_values[u_idx] = u
        fcn.set_input_values(input_values)

        hessian = fcn.evaluate_hessian_outputs()
        pred_hess = [
            [[2., 0.], [0., 4.]],
            [[0., 1.], [1., 0.]],
            ]
        for pred, act in zip(pred_hess, hessian):
            np.testing.assert_allclose(pred, act.toarray())


class TestComposeFunctionFromNLP(unittest.TestCase):

    def test_outputs(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        self.assertEqual(to_embed.n_inputs(), 3)
        self.assertEqual(to_embed.n_outputs(), 2)

        y, z, u = 2.0, 3.0, 4.0
        # Need a way to map my inputs to their coordinates
        # I.e. need the offset
        y_idx_f = to_embed.get_input_offset(x_idx)
        z_idx_f = y_idx_f + 1
        u_idx_f = to_embed.get_input_offset(u_idx)
        inputs = np.zeros(3)
        inputs[y_idx_f] = y
        inputs[z_idx_f] = z
        inputs[u_idx_f] = u

        to_embed.set_input_values(inputs)
        intermed_outputs = to_embed.evaluate_outputs()
        # Now need a way to get output offsets.
        # Easy here because my outputs are one-dimensional
        pred_outputs = [None, None]
        # These indices could be different than x_idx and u_idx.
        # x_idx and u_idx are just indices into the user-provided
        # list of functions.
        # These indices are actual inputs into the nlp.
        x_idx_nlp = to_embed.get_output_offset(x_idx)
        u_idx_nlp = to_embed.get_output_offset(u_idx)
        pred_outputs[x_idx] = np.sqrt(y**2 + z**2)
        pred_outputs[u_idx] = u
        np.testing.assert_allclose(intermed_outputs, pred_outputs)

        fcn_comp = FunctionComposition(nlp_fcn, to_embed)
        fcn_comp.set_input_values(inputs)
        nlp_outputs = fcn_comp.evaluate_outputs()
        # I know these coordinates because my NLP only has two
        # "row coordinates." If it had multiple constraint functions,
        # I would use get_constraint_indices.
        pred_outputs = [
            y**2 + z**2 + 2*u**2,
            u*np.sqrt(y**2 + z**2) - 1.0,
        ]
        np.testing.assert_allclose(nlp_outputs, pred_outputs)

    def test_jacobian(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        # Is this really the best way order the functions I need to
        # send to the NLP?
        # ... Ideally I re-order the NLP the match some partition
        # in my model...
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        y, z, u = 2.0, 3.0, 4.0
        inputs = np.array([y, z, u])
        to_embed.set_input_values(inputs)
        intermed_jac = to_embed.evaluate_jacobian_outputs()
        # Now need a way to get output offsets.
        # Easy here because my outputs are one-dimensional
        pred_jac = np.zeros((2, 3))
        y_idx_f = to_embed.get_input_offset(0)
        z_idx_f = y_idx_f + 1
        u_idx_f = to_embed.get_input_offset(1)
        denom = np.sqrt(y**2 + z**2)
        pred_jac[x_idx][y_idx_f] = y/denom
        pred_jac[x_idx][z_idx_f] = z/denom
        pred_jac[u_idx][u_idx_f] = 1.0
        np.testing.assert_allclose(intermed_jac.toarray(), pred_jac)

        fcn_comp.set_input_values(inputs)
        jacobian = fcn_comp.evaluate_jacobian_outputs()
        pred_jac = np.zeros((2, 3))
        denom = np.sqrt(y**2 + z**2)
        pred_jac[0, y_idx_f] = 2*y
        pred_jac[0, z_idx_f] = 2*z
        pred_jac[0, u_idx_f] = 4*u
        pred_jac[1, y_idx_f] = u*y/denom
        pred_jac[1, z_idx_f] = u*z/denom
        pred_jac[1, u_idx_f] = denom
        np.testing.assert_allclose(jacobian.toarray(), pred_jac)

    def test_hessian(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        y, z, u = 2.0, 3.0, 4.0
        inputs = np.array([y, z, u])
        to_embed.set_input_values(inputs)
        intermed_hess = to_embed.evaluate_hessian_outputs()

        pred_hess = np.zeros((2, 3, 3))
        y_idx_f = to_embed.get_input_offset(0)
        z_idx_f = y_idx_f + 1
        u_idx_f = to_embed.get_input_offset(1)
        denom = np.sqrt(y**2 + z**2)
        pred_hess[x_idx, y_idx_f, y_idx_f] = z**2/denom**3
        pred_hess[x_idx, y_idx_f, z_idx_f] = -y*z/denom**3
        pred_hess[x_idx, z_idx_f, y_idx_f] = -y*z/denom**3
        pred_hess[x_idx, z_idx_f, z_idx_f] = y**2/denom**3
        # pred_hess[u_idx, ...] is all zeros

        for pred, act in zip(pred_hess, intermed_hess):
            np.testing.assert_allclose(pred, act.toarray())

        fcn_comp.set_input_values(inputs)
        hessian = fcn_comp.evaluate_hessian_outputs()
        pred_hess = np.zeros((2, 3, 3))
        pred_hess[0, y_idx_f, y_idx_f] = 2.0
        pred_hess[0, z_idx_f, z_idx_f] = 2.0
        pred_hess[0, u_idx_f, u_idx_f] = 4.0
        pred_hess[1, y_idx_f, y_idx_f] = u*z**2/denom**3
        pred_hess[1, y_idx_f, z_idx_f] = -u*z*y/denom**3
        pred_hess[1, z_idx_f, y_idx_f] = -u*z*y/denom**3
        pred_hess[1, z_idx_f, z_idx_f] = u*y**2/denom**3
        pred_hess[1, z_idx_f, u_idx_f] = z/denom
        pred_hess[1, u_idx_f, z_idx_f] = z/denom
        pred_hess[1, y_idx_f, u_idx_f] = y/denom
        pred_hess[1, u_idx_f, y_idx_f] = y/denom

        # NOTE: These individual hessians are more dense than they need
        # to be because ASL returns matrices with the nonzero structure
        # of the entire Hessian-of-Lagrangian.
        self.assertEqual(hessian[0].nnz, 9)
        self.assertEqual(hessian[1].nnz, 9)
        #self.assertEqual(hessian[0].nnz, 5)
        #self.assertEqual(hessian[1].nnz, 8)

        for pred, act in zip(pred_hess, hessian):
            # Need nonzero atol here because of numerical cancelation in the
            # calculation of the objective Hessian.
            np.testing.assert_allclose(pred, act.toarray(), atol=1e-15)


class TestNLPFromFunction(unittest.TestCase):

    def test_primals(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        nlp = NLPFromFunction(fcn_comp)
        self.assertEqual(nlp.n_primals(), 3)

    def test_constraints(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        nlp = NLPFromFunction(fcn_comp)
        self.assertEqual(nlp.n_constraints(), 1)
        np.testing.assert_array_equal(nlp.constraints_lb(), [0.0])
        np.testing.assert_array_equal(nlp.constraints_ub(), [0.0])

    @unittest.skipUnless(cyipopt_available, "CyIpopt is not available")
    def test_cyipoptnlp(self):
        m_yz = make_model1_yzu()
        solver = pyo.SolverFactory("ipopt")
        solver.solve(m_yz, tee=True)

        m = make_model1_xu()
        pyomo_nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(pyomo_nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = pyomo_nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        x0 = np.zeros(3)
        x_idx_f = to_embed.get_input_offset(x_idx)
        u_idx_f = to_embed.get_input_offset(u_idx)
        x0[x_idx_f] = 1.5
        x0[x_idx_f + 1] = 1.5
        x0[u_idx_f] = 2.5

        nlp = NLPFromFunction(fcn_comp)
        problem = CyIpoptNLP(nlp)
        cyipopt = CyIpoptSolver(problem)
        x, res = cyipopt.solve(x0=x0, tee=True)

        self.assertAlmostEqual(x[x_idx_f], m_yz.y.value)
        self.assertAlmostEqual(x[x_idx_f + 1], m_yz.z.value)
        self.assertAlmostEqual(x[u_idx_f], m_yz.u.value)


class TestEmbedFunctionWithExistingInputs(unittest.TestCase):

    def _create_model_and_nlp(self):
        m = make_model2_xyzu()
        fcn = SimpleFunction2()

        pyomo_nlp = PyomoNLP(m)
        iden_u = IdentityFunction(1)
        iden_yz = IdentityFunction(2)

        # Create a function mapping (y, z) -> (x, y, z)
        fcn_yz = FunctionCombination(fcn, iden_yz)
        # Create a function mapping (y, z, u) -> (x, y, z, u)
        # This is the function we want to embed in our NLP
        fcn_yzu = FunctionStack(fcn_yz, iden_u)

        # Need primals of the PyomoNLP to be in the order (x, y, z, u)
        ix, iy, iz, iu = pyomo_nlp.get_primal_indices([m.x, m.y, m.z, m.u])
        names = pyomo_nlp.primals_names()
        # Unfortunate that names are used here rather than coordinates
        primals_ordering = [names[ix], names[iy], names[iz], names[iu]]
        proj_nlp = ProjectedNLP(pyomo_nlp, primals_ordering)
        # Create a function from our reordered NLP
        nlp_fcn = FunctionFromNLP(proj_nlp)

        # Compose function-from-NLP with the new function mapping
        # "embedded variables" to "NLP variables"
        fcn_comp = FunctionComposition(nlp_fcn, fcn_yzu)
        nlp = NLPFromFunction(fcn_comp)

        return m, pyomo_nlp, nlp

    def test_nlp_constraints(self):
        m, pyomo_nlp, nlp = self._create_model_and_nlp()
        # Because I created the "inner function" and re-ordered the NLP,
        # I know that the coordinates correspond to (y, z, u) in order
        x0 = np.array([1.5, 1.5, 2.5])
        m.x.set_value(pyo.sqrt(1.5**2 + 1.5**2))

        self.assertEqual(nlp.n_primals(), 3)
        self.assertEqual(nlp.n_constraints(), 2)

        nlp.set_primals(x0)

        clb1 = nlp.constraints_lb()
        clb2 = pyomo_nlp.constraints_lb()
        cub1 = nlp.constraints_ub()
        cub2 = pyomo_nlp.constraints_ub()
        np.testing.assert_array_equal(clb1, clb2)
        np.testing.assert_array_equal(cub1, cub2)

        constraints = pyomo_nlp.get_pyomo_constraints()
        con_values = nlp.evaluate_constraints()
        for con, val in zip(constraints, con_values):
            self.assertEqual(pyo.value(con.body), val)

    def test_nlp_jacobian(self):
        m, pyomo_nlp, nlp = self._create_model_and_nlp()
        x0 = np.array([1.5, 1.5, 2.5])
        nlp.set_primals(x0)
        iy, iz, iu = 0, 1, 2
        j1, j2 = pyomo_nlp.get_constraint_indices([m.eq_con1, m.eq_con2])

        jac = nlp.evaluate_jacobian()

        M = nlp.n_constraints()
        N = nlp.n_primals()

        rcd = []
        denom = pyo.sqrt(m.y.value**2 + m.z.value**2)
        rcd.append((j1, iy, pyo.value(m.y*m.u/denom)))
        rcd.append((j1, iz, pyo.value(m.z*m.u/denom)))
        rcd.append((j1, iu, denom))
        rcd.append((j2, iy, m.z.value))
        rcd.append((j2, iz, m.y.value))
        row = [r for r, _, _ in rcd]
        col = [c for _, c, _ in rcd]
        data = [d for _, _, d in rcd]
        pred_jac = sps.coo_matrix((data, (row, col)), shape=(M, N))

        self.assertEqual(pred_jac.nnz, jac.nnz)
        nz_set = set((i, j) for i, j, d in rcd)
        for i, j in zip(jac.row, jac.col):
            self.assertIn((i, j), nz_set)
        np.testing.assert_allclose(jac.toarray(), pred_jac.toarray())

    def test_nlp_objective(self):
        m, pyomo_nlp, nlp = self._create_model_and_nlp()
        x0 = np.array([1.5, 1.5, 2.5])
        nlp.set_primals(x0)
        iy, iz, iu = 0, 1, 2
        m.x.set_value(pyo.sqrt(1.5**2 + 1.5**2))

        obj_val = nlp.evaluate_objective()
        self.assertEqual(pyo.value(m.obj), obj_val)

        grad = nlp.evaluate_grad_objective()
        pred_grad = [pyo.value(2*m.y), pyo.value(2*m.z), pyo.value(4*m.u)]
        np.testing.assert_allclose(grad, pred_grad)

    def test_nlp_hessian(self):
        m, pyomo_nlp, nlp = self._create_model_and_nlp()
        x0 = np.array([1.5, 1.5, 2.5])
        nlp.set_primals(x0)
        iy, iz, iu = 0, 1, 2
        j1, j2 = pyomo_nlp.get_constraint_indices([m.eq_con1, m.eq_con2])
        N = nlp.n_primals()

        duals = np.array([1.1, 2.2])
        obj_factor = 0.5
        nlp.set_duals(duals)
        nlp.set_obj_factor(obj_factor)
        hess = nlp.evaluate_hessian_lag()

        rcd0 = []
        rcd0.append((iy, iy, 2.0))
        rcd0.append((iz, iz, 2.0))
        rcd0.append((iu, iu, 4.0))
        row = [r for r, _, _ in rcd0]
        col = [c for _, c, _ in rcd0]
        data = [d for _, _, d in rcd0]
        obj_hess = sps.coo_matrix((data, (row, col)), shape=(N, N))

        denom = pyo.value(pyo.sqrt(m.y**2 + m.z**2))
        rcd1 = []
        rcd1.append((iy, iy, pyo.value(m.u*(denom**2 - m.y**2)/denom**3)))
        rcd1.append((iz, iz, pyo.value(m.u*(denom**2 - m.z**2)/denom**3)))
        rcd1.append((iy, iz, pyo.value(-m.y*m.z*m.u/denom**3)))
        rcd1.append((iz, iy, pyo.value(-m.y*m.z*m.u/denom**3)))
        rcd1.append((iy, iu, pyo.value(m.y/denom)))
        rcd1.append((iu, iy, pyo.value(m.y/denom)))
        rcd1.append((iz, iu, pyo.value(m.z/denom)))
        rcd1.append((iu, iz, pyo.value(m.z/denom)))
        row = [r for r, _, _ in rcd1]
        col = [c for _, c, _ in rcd1]
        data = [d for _, _, d in rcd1]
        hess1 = sps.coo_matrix((data, (row, col)), shape=(N, N))

        rcd2 = []
        rcd2.append((iy, iz, 1.0))
        rcd2.append((iz, iy, 1.0))
        row = [r for r, _, _ in rcd2]
        col = [c for _, c, _ in rcd2]
        data = [d for _, _, d in rcd2]
        hess2 = sps.coo_matrix((data, (row, col)), shape=(N, N))

        pred_hess = obj_factor*obj_hess + duals[0]*hess1 + duals[1]*hess2
        pred_hess = pred_hess.tocoo()

        self.assertEqual(pred_hess.nnz, hess.nnz)
        nz_set = set(zip(pred_hess.row, pred_hess.col))
        for i, j in zip(hess.row, hess.col):
            self.assertIn((i, j), nz_set)
        np.testing.assert_allclose(pred_hess.toarray(), hess.toarray())

    @unittest.skipUnless(cyipopt_available, "CyIpopt is not available")
    def test_solve(self):
        m, pyomo_nlp, nlp = self._create_model_and_nlp()
        m_yzu = make_model2_yzu()

        x0 = np.array([1.5, 1.5, 2.5])
        nlp.set_primals(x0)

        problem = CyIpoptNLP(nlp)
        cyipopt = CyIpoptSolver(problem)
        x, res = cyipopt.solve(x0=x0)

        ipopt = pyo.SolverFactory("ipopt")
        ipopt.solve(m_yzu)
        pred_values = [m_yzu.y.value, m_yzu.z.value, m_yzu.u.value]

        np.testing.assert_allclose(pred_values, x)


class TestNLPFromFunctionFromNLP(unittest.TestCase):
    def _make_qp_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], initialize=1.0)
        m.con = pyo.Constraint(
            [1, 2],
            rule={
                1: 0.05*m.x[1] + m.x[2] <= 2,
                2: m.x[1] + 0.2*m.x[2] <= 2,
            },
        )
        m.obj = pyo.Objective(expr=(m.x[1] - 2)**2 + (m.x[2] - 2)**2)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        return m

    def _make_nonlin_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], initialize=0.0, bounds=(0, None))
        m.con = pyo.Constraint(
            [1, 2],
            rule={
                1: m.x[1]**2 + m.x[2] - 2 == 0,
                2: m.x[1] - m.x[2] >= 0,
            },
        )
        m.obj = pyo.Objective(expr=(m.x[1] - 2)**2 + (m.x[2] - 2)**2)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        return m

    def _make_square_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.0)
        m.u = pyo.Var(initialize=1.0)
        m.con = pyo.Constraint(expr=m.x*m.u - 3.0 == 0)
        m.u_con = pyo.Constraint(expr=m.u - 2.0 == 0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        m.obj = pyo.Objective(expr=0.0)
        return m

    @unittest.skipUnless(cyipopt_available, "CyIpopt is not available")
    def test_qp(self):
        m = self._make_qp_model()
        nlp = PyomoNLP(m)
        fcn = FunctionFromNLP(nlp)
        fcn_nlp = NLPFromFunction(fcn)

        solver = pyo.SolverFactory("ipopt")
        res_pyomo = solver.solve(m)

        x0 = np.array([1.0, 1.0])
        problem = CyIpoptNLP(fcn_nlp)
        cyipopt = CyIpoptSolver(problem)
        x, res_nlp = cyipopt.solve(x0=x0)

        x1_idx, x2_idx = nlp.get_primal_indices([m.x[1], m.x[2]])
        c1_idx, c2_idx = nlp.get_constraint_indices([m.con[1], m.con[2]])
        primal_pyomo = [None, None]
        primal_pyomo[x1_idx] = m.x[1].value
        primal_pyomo[x2_idx] = m.x[2].value
        dual_pyomo = [None, None]
        dual_pyomo[c1_idx] = m.dual[m.con[1]]
        dual_pyomo[c2_idx] = m.dual[m.con[2]]

        dual_nlp = res_nlp['mult_g']

        np.testing.assert_allclose(x, primal_pyomo)
        # Ipopt Ampl interface and direct interface have
        # different conventions for multipliers.
        np.testing.assert_allclose(-dual_nlp, dual_pyomo)

    @unittest.skipUnless(cyipopt_available, "CyIpopt is not available")
    def test_nonlin(self):
        m = self._make_nonlin_model()
        nlp = PyomoNLP(m)
        fcn = FunctionFromNLP(nlp)
        fcn_nlp = NLPFromFunction(fcn)

        solver = pyo.SolverFactory("ipopt")
        res_pyomo = solver.solve(m)

        x0 = np.array([0.0, 0.0])
        problem = CyIpoptNLP(fcn_nlp)
        cyipopt = CyIpoptSolver(problem)
        x, res_nlp = cyipopt.solve(x0=x0)

        x1_idx, x2_idx = nlp.get_primal_indices([m.x[1], m.x[2]])
        c1_idx, c2_idx = nlp.get_constraint_indices([m.con[1], m.con[2]])
        primal_pyomo = [None, None]
        primal_pyomo[x1_idx] = m.x[1].value
        primal_pyomo[x2_idx] = m.x[2].value
        dual_pyomo = [None, None]
        dual_pyomo[c1_idx] = m.dual[m.con[1]]
        dual_pyomo[c2_idx] = m.dual[m.con[2]]

        dual_nlp = res_nlp['mult_g']

        np.testing.assert_allclose(x, primal_pyomo)
        # Ipopt Ampl interface and direct interface have
        # different conventions for multipliers.
        np.testing.assert_allclose(-dual_nlp, dual_pyomo)

    @unittest.skipUnless(cyipopt_available, "CyIpopt is not available")
    def test_square(self):
        m = self._make_square_model()
        nlp = PyomoNLP(m)
        fcn = FunctionFromNLP(nlp)
        fcn_nlp = NLPFromFunction(fcn)

        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        c1_idx, c2_idx = nlp.get_constraint_indices([m.con, m.u_con])

        solver = pyo.SolverFactory("ipopt")
        res_pyomo = solver.solve(m)

        primal_pyomo = [None, None]
        primal_pyomo[x_idx] = m.x.value
        primal_pyomo[u_idx] = m.u.value
        dual_pyomo = [None, None]
        dual_pyomo[c1_idx] = m.dual[m.con]
        dual_pyomo[c2_idx] = m.dual[m.u_con]

        x0 = np.zeros(nlp.n_primals())
        x0[x_idx] = 0.0
        x0[u_idx] = 1.0

        problem = CyIpoptNLP(fcn_nlp)
        cyipopt = CyIpoptSolver(problem)
        x, res_nlp = cyipopt.solve(x0=x0)
        dual_nlp = res_nlp["mult_g"]

        np.testing.assert_allclose(x, primal_pyomo)
        np.testing.assert_allclose(-dual_nlp, dual_pyomo)
        # Nondegenerate square problem should have multipliers of zero.
        np.testing.assert_allclose(-dual_nlp, [0.0, 0.0])


class TestReFixVars(unittest.TestCase):
    """
    This tests the functionality of using an NLP to "fix" variables
    by adding equality constraints.

    """

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], initialize=5.0)
        m.u = pyo.Var(initialize=1.0)
        m.eq_con = pyo.Constraint(expr=m.x[1]*m.x[2] - m.u == 0.0)
        m.obj = pyo.Objective(expr=m.x[1]**2 + 3.0*m.x[2]**2)
        return m

    @unittest.skipUnless(cyipopt_available, "CyIpopt is not available")
    def test_solve_fix_constraints(self):
        n_primals = 3
        value_map = {0: 1, 1: 2, 2: 3}
        fixing_constraints = FixedVarNLP(n_primals, value_map)

        problem = CyIpoptNLP(fixing_constraints)
        cyipopt = CyIpoptSolver(problem)
        x, results = cyipopt.solve(tee=True)
        np.testing.assert_allclose(x, [1, 2, 3])

    @unittest.skipUnless(cyipopt_available, "CyIpopt is not available")
    def test_solve_fixed_nlp(self):
        m_fixed = self._make_model()
        m_fixed.u.fix(2.0)
        solver = pyo.SolverFactory("ipopt")
        solver.solve(m_fixed, tee=True)

        m = self._make_model()
        nlp = PyomoNLP(m)
        pyomo_vars = [m.x[1], m.x[2], m.u]
        x1_idx, x2_idx, u_idx = nlp.get_primal_indices(pyomo_vars)
        n_primals = nlp.n_primals()
        value_map = {u_idx: 2.0}
        fixing_constraints = FixedVarNLP(n_primals, value_map)

        nlp_fcn = FunctionFromNLP(nlp)
        fixing_fcn = FunctionFromNLP(
            fixing_constraints, include_objective=False
        )

        combined_fcn = FunctionCombination(nlp_fcn, fixing_fcn)
        combined_nlp = NLPFromFunction(combined_fcn)

        x0 = nlp.get_primals()
        x0 = np.array([5.0, 5.0, 1.0])
        combined_nlp.set_primals(x0)

        problem = CyIpoptNLP(combined_nlp)
        cyipopt = CyIpoptSolver(problem)
        x, res = cyipopt.solve(x0=x0, tee=True)

        self.assertAlmostEqual(x[x1_idx], m_fixed.x[1].value)
        self.assertAlmostEqual(x[x2_idx], m_fixed.x[2].value)
        self.assertAlmostEqual(x[u_idx], m_fixed.u.value)

    @unittest.skipUnless(cyipopt_available, "CyIpopt is not available")
    def test_refix_and_solve(self):
        m_fixed = self._make_model()
        m_fixed.u.fix(2.0)
        solver = pyo.SolverFactory("ipopt")
        solver.solve(m_fixed, tee=True)

        m = self._make_model()
        nlp = PyomoNLP(m)
        pyomo_vars = [m.x[1], m.x[2], m.u]
        x1_idx, x2_idx, u_idx = nlp.get_primal_indices(pyomo_vars)
        n_primals = nlp.n_primals()
        value_map = {u_idx: 2.0}
        fixing_constraints = FixedVarNLP(n_primals, value_map)

        nlp_fcn = FunctionFromNLP(nlp)
        fixing_fcn = FunctionFromNLP(
            fixing_constraints, include_objective=False
        )

        combined_fcn = FunctionCombination(nlp_fcn, fixing_fcn)
        combined_nlp = NLPFromFunction(combined_fcn)

        x0 = nlp.get_primals()
        x0 = np.array([5.0, 5.0, 1.0])
        combined_nlp.set_primals(x0)

        problem = CyIpoptNLP(combined_nlp)
        cyipopt = CyIpoptSolver(problem)
        x, res = cyipopt.solve(x0=x0, tee=True)

        self.assertAlmostEqual(x[x1_idx], m_fixed.x[1].value)
        self.assertAlmostEqual(x[x2_idx], m_fixed.x[2].value)
        self.assertAlmostEqual(x[u_idx], m_fixed.u.value)

        m_fixed.u.fix(3.0)
        solver.solve(m_fixed, tee=True)

        value_map = {u_idx: 3.0}
        fixing_constraints.update_fixed_values(value_map)
        x0 = x
        x, res = cyipopt.solve(x0=x0, tee=True)
        self.assertAlmostEqual(x[x1_idx], m_fixed.x[1].value)
        self.assertAlmostEqual(x[x2_idx], m_fixed.x[2].value)
        self.assertAlmostEqual(x[u_idx], m_fixed.u.value)

    @unittest.skipUnless(cyipopt_available, "CyIpopt is not available")
    def test_fixed_nlp(self):
        m = self._make_model()
        nlp = PyomoNLP(m)
        pyomo_vars = [m.x[1], m.x[2], m.u]
        x1_idx, x2_idx, u_idx = nlp.get_primal_indices(pyomo_vars)
        n_primals = nlp.n_primals()
        value_map = {u_idx: 2.0}
        fixing_constraints = FixedVarNLP(n_primals, value_map)

        nlp_fcn = FunctionFromNLP(nlp)
        fixing_fcn = FunctionFromNLP(
            fixing_constraints, include_objective=False
        )

        combined_fcn = FunctionCombination(nlp_fcn, fixing_fcn)
        combined_nlp = NLPFromFunction(combined_fcn)
        self.assertEqual(combined_nlp.n_primals(), 3)
        self.assertEqual(combined_nlp.n_constraints(), 2)

        primals = nlp.get_primals()
        # This is necessary as I don't yet automatically set
        # init primals of an NLP-from-function
        combined_nlp.set_primals(primals)

        model_con_idx = nlp.get_constraint_indices([m.eq_con])[0]
        # I happen to know that the "fixing constraints" come after
        # the "model constraints"
        fix_con_idx = nlp.n_constraints()

        resid = combined_nlp.evaluate_constraints()
        pred_resid = np.zeros(2)
        pred_resid[model_con_idx] = pyo.value(m.eq_con.body)
        pred_resid[fix_con_idx] = m.u.value - 2.0
        np.testing.assert_allclose(resid, pred_resid)

        jac = combined_nlp.evaluate_jacobian()
        row = []
        col = []
        data = []

        row.append(model_con_idx)
        col.append(x1_idx)
        data.append(primals[x2_idx])

        row.append(model_con_idx)
        col.append(x2_idx)
        data.append(primals[x1_idx])

        row.append(model_con_idx)
        col.append(u_idx)
        data.append(-1.0)

        row.append(fix_con_idx)
        col.append(u_idx)
        data.append(1.0)

        pred_jac = sps.coo_matrix((data, (row, col)), shape=(2, 3))
        np.testing.assert_allclose(jac.toarray(), pred_jac.toarray())

        grad_obj = combined_nlp.evaluate_grad_objective()
        pred_grad = np.zeros(combined_nlp.n_primals())
        pred_grad[x1_idx] = 2.0*primals[x1_idx]
        pred_grad[x2_idx] = 6.0*primals[x2_idx]
        np.testing.assert_allclose(grad_obj, pred_grad)

        combined_nlp.set_duals(np.ones(combined_nlp.n_constraints()))
        hess = combined_nlp.evaluate_hessian_lag()
        row = []
        col = []
        data = []

        row.append(x1_idx)
        col.append(x2_idx)
        data.append(1.0)

        row.append(x2_idx)
        col.append(x1_idx)
        data.append(1.0)
        con_hess = sps.coo_matrix((data, (row, col)), shape=(3, 3))

        row = []
        col = []
        data = []
        row.append(x1_idx)
        col.append(x1_idx)
        data.append(2.0)

        row.append(x2_idx)
        col.append(x2_idx)
        data.append(6.0)
        obj_hess = sps.coo_matrix((data, (row, col)), shape=(3, 3))
        pred_hess = con_hess + obj_hess
        np.testing.assert_allclose(hess.toarray(), pred_hess.toarray())


if __name__ == '__main__':
    unittest.main()
