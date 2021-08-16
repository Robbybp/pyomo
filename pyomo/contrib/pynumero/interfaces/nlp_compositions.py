from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    VectorValuedExternalFunction,
)
import numpy as np
import scipy.sparse as sps


class FunctionComposition(VectorValuedExternalFunction):

    def __init__(self, function1, function2, coord_match=None):
        """
        function1(function2(...))

        Parameters
        ----------
        function1: VectorValuedExternalFunction
            The first (outer) function in the composition
        function2: VectorValuedExternalFunction
            The second (inner) function in the composition
        coord_match: Iterable
            For each output coordinate of the inner function, the
            corresponding input coordinate of the outer function
                
        """
        # TODO: Should probably support
        #     function1.n_inputs > function2.n_outputs()
        assert function1.n_inputs() == function2.n_outputs()

        self._function1 = function1
        self._function2 = function2

    def n_inputs(self):
        return self._function2.n_inputs()

    def n_outputs(self):
        return self._function1.n_outputs()

    def set_input_values(self, input_values):
        self._function2.set_input_values(input_values)
        ouputs2 = self._function2.evaluate_outputs()
        # TODO: permute input values if necessary
        self._function1.set_input_values(outputs2)

    def evaluate_outputs(self):
        return self._function1.evaluate_outputs()

    def evaluate_jacobian_outputs(self):
        pass

    def evaluate_hessian_outputs(self):
        pass
