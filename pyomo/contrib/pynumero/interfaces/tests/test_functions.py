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
    IdentityFunction,
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


class TestComposeFunctionFromNLP(self):

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


if __name__ == '__main__':
    unittest.main()
