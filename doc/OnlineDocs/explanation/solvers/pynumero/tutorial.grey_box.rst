Embedding vector-valued external functions with ``ExternalGreyBoxBlock``
========================================================================

Scalar-valued external functions may be embedded into Pyomo models using the AMPL
external function interface. See :ref:`aslfunctions` for several examples.
However, there are times when you might want to embed a single vector-valued
function rather than many individual scalar-valued functions:

- You can evaluate many outputs simultaneously in a single function call. (This *could*
  be implemented with many individual scalar-valued functions as long
  as they have a shared cache.)
- You can evaluate the Hessian of the Lagrangian of your vector-valued function
  (defined as the multiplier-weighted sum of individual output Hessians)
  more efficiently than if you evaluated every output's Hessian matrix.
- You would like to bypass the AMPL external function infrastructure (which
  requires writing your function in C and compiling it into a shared library)
  and instead write your external function in Python.

PyNumero provides users the ability to embed vector-valued external functions into
Pyomo models using the
:class:`ExternalGreyBoxBlock<pyomo.contrib.pynumero.interfaces.external_grey_box.ExternalGreyBoxBlock>`
modeling component.
We add these functions into a model as constraints of the form :math:`y = F(x)`
or :math:`G(x) = 0`.
These models may then be solved with the ``cyipopt`` solver.

Step 0: Imports
---------------

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> import pyomo.environ as pyo
   >>> import numpy as np
   >>> import scipy.sparse as sps
   >>> from pyomo.contrib.pynumero.interfaces.external_grey_box import (
   ...    ExternalGreyBoxModel,
   ...    ExternalGreyBoxBlock,
   ... )

Step 1: Implement your function as an ``ExternalGreyBoxModel``
--------------------------------------------------------------

We would like to embed the function,

.. math::

   y = x_0 - x_2 + 4 \\
   x_0 x_1^{1.5} x_2 - 5 = 0 \\
   (x_0 - x_2 + 4)e^{-x_1} - 1 = 0, \\

where :math:`y` is an output and :math:`x_0`, :math:`x_1`, and :math:`x_2` are inputs.
We will implement this function as an external grey box with three inputs, one output,
and two equality constraints.

First, we must implement our function as an
:class:`ExternalGreyBoxModel<pyomo.contrib.pynumero.interfaces.external_grey_box.ExternalGreyBoxModel>`.
We do this by subclassing this base class and implementing methods for function, Jacobian, and, optionally,
Hessian evaluation. The new class will define our external function only; we are not yet connecting
this function to any Pyomo modeling components.

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> class MyGreyBox(ExternalGreyBoxModel):
   ... 
   ...     def __init__(self):
   ...         self._input_values = np.zeros(self.n_inputs(), dtype=float)
   ...         self._equality_constraint_multiplier_values = np.zeros(
   ...             self.n_equality_constraints(), dtype=float
   ...         )
   ...         self._output_constraint_multiplier_values = np.zeros(
   ...             self.n_outputs(), dtype=float
   ...         )
   ... 
   ...     def input_names(self):
   ...         return ["x[0]", "x[1]", "x[2]"]
   ... 
   ...     def output_names(self):
   ...         return ["y"]
   ... 
   ...     def equality_constraint_names(self):
   ...         return ["eq[0]", "eq[1]"]
   ... 
   ...     def set_input_values(self, input_values):
   ...         assert len(input_values) == self.n_inputs()
   ...         self._input_values = np.asarray(input_values, dtype=float)
   ... 
   ...     def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
   ...         assert len(eq_con_multiplier_values) == self.n_equality_constraints()
   ...         np.copyto(
   ...             self._equality_constraint_multiplier_values,
   ...             np.asarray(eq_con_multiplier_values, dtype=float),
   ...         )
   ... 
   ...     def set_output_constraint_multipliers(self, output_con_multiplier_values):
   ...         assert len(output_con_multiplier_values) == self.n_outputs()
   ...         np.copyto(
   ...             self._output_constraint_multiplier_values,
   ...             np.asarray(output_con_multiplier_values, dtype=float),
   ...         )
   ... 
   ...     def evaluate_outputs(self):
   ...         x = self._input_values
   ...         return np.array([x[0] - x[2] + 4.0])
   ... 
   ...     def evaluate_equality_constraints(self):
   ...         x = self._input_values
   ...         return np.array([
   ...             x[0] * x[1]**1.5 * x[2] - 5.0,
   ...             (x[0] - x[2] + 4.0) * np.exp(-x[1]) - 1.0
   ...         ])
   ... 
   ...     def evaluate_jacobian_outputs(self):
   ...         rows = np.array([0, 0], dtype=int)
   ...         cols = np.array([0, 2], dtype=int)
   ...         data = np.array([1.0, -1.0], dtype=float)
   ...         return coo_matrix((data, (rows, cols)), shape=(self.n_outputs(), self.n_inputs()))
   ... 
   ...     def evaluate_jacobian_equality_constraints(self):
   ...         x = self._input_values
   ...         exp_neg_x1 = np.exp(-x[1])
   ...         rows = np.array([0, 0, 0, 1, 1, 1], dtype=int)
   ...         cols = np.array([0, 1, 2, 0, 1, 2], dtype=int)
   ...         data = np.array(
   ...             [
   ...                 x[1] ** 1.5 * x[2],
   ...                 1.5 * x[0] * x[1] ** 0.5 * x[2],
   ...                 x[0] * x[1] ** 1.5,
   ...                 exp_neg_x1,
   ...                 -(x[0] - x[2] + 4.0) * exp_neg_x1,
   ...                 -exp_neg_x1,
   ...             ],
   ...             dtype=float,
   ...         )
   ...         return coo_matrix(
   ...             (data, (rows, cols)),
   ...             shape=(self.n_equality_constraints(), self.n_inputs()),
   ...         )
   ... 
   ...     def evaluate_hessian_equality_constraints(self):
   ...         x = self._input_values
   ...         lam = self._equality_constraint_multiplier_values
   ...         rows = np.array([1, 2, 1, 2, 1, 2], dtype=int)
   ...         cols = np.array([0, 0, 1, 1, 0, 1], dtype=int)
   ...         data = np.array(
   ...             [
   ...                 lam[0] * 1.5 * x[1] ** 0.5 * x[2] - lam[1] * np.exp(-x[1]),
   ...                 lam[0] * x[1] ** 1.5,
   ...                 (
   ...                     lam[0] * 0.75 * x[0] * x[1] ** (-0.5) * x[2]
   ...                     + lam[1] * (x[0] - x[2] + 4.0) * np.exp(-x[1])
   ...                 ),
   ...                 lam[0] * 1.5 * x[0] * x[1] ** 0.5,
   ...                 0.0,
   ...                 lam[1] * np.exp(-x[1]),
   ...             ],
   ...             dtype=float,
   ...         )
   ...         return coo_matrix((data, (rows, cols)), shape=(self.n_inputs(), self.n_inputs()))
   ... 
   ...     def evaluate_hessian_outputs(self):
   ...         return coo_matrix((self.n_inputs(), self.n_inputs()))

``ExternalGreyBoxModel`` implements a "stateful" model of these functions.
We set inputs or Lagrange multipliers with one of the `set_*` methods,
then use these values for any subsequent calculations. The work of actually
evaluating the functions may be performed at the time the inputs are set
or the time the outputs are evaluated, depending on what is convenient
for the target application.

Some additional notes on the above:

- Note that ``input_names`` and ``output_names`` are required to be implemented
  in derived classes. The ``n_inputs`` and ``n_outputs`` methods use the lists
  returned by the ``*_name`` methods.
- The Jacobian and Hessian matrices should contain entries for all matrices that
  *can possibly* be nonzero. These sparsity structures are assumed to not change
  between different evaluations.
- The ``evaluate_hessian_equality_constraints`` and ``evaluate_hessian_outputs`` methods are
  misnamed. Instead of returning the second-derivative tensors of the vector-valued
  :math:`F` and :math:`G` functions, they return the term in the Hessian of the
  Lagrangian corresponding to these functions. That is, they return the sum of the
  Hessians of the individual output (or equality constraint) coordinates, weighted
  by the Lagrange multiplier of that output. That is, the
  ``evaluate_hessian_outputs`` method returns:

.. math::

   \sum_{i=1}^{n_y} \lambda_{F,i}\nabla^2 F_i

For the full documentation of these methods, please see the
:class:`ExternalGreyBoxModel class documentation<pyomo.contrib.pynumero.interfaces.external_grey_box.ExternalGreyBoxModel>`.

Step 2: Construct an ``ExternalGreyBoxBlock``
---------------------------------------------

Now that we have constructed our external function, it is time to embed
it into an optimization problem. First, we create a Pyomo model:

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> m = pyo.ConcreteModel()

When adding the external function to the model, we have the choice between
adding the input and output variables ourselves, or letting the ``ExternalGreyBoxBlock``
add them for us automatically. If we let the grey box block add the
variables, we will need to appropriately link them with the rest of the model.
Here, we will define our own input variables, but let the grey box block define
the output variable for us.

Step 3: Solve the model with CyIpopt
------------------------------------
