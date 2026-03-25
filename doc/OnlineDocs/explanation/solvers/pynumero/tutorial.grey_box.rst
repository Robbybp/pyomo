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
modeling component. These models may then be solved with the ``cyipopt`` solver.

Step 0: Imports
---------------

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

   >>> import pyomo.environ as pyo
   >>> from pyomo.contrib.pynumero.interfaces.external_grey_box import (
          ExternalGreyBoxModel,
          ExternalGreyBoxBlock,
       )

Step 1: Implement your function as an ``ExternalGreyBoxModel``
--------------------------------------------------------------

First, we implement an
:class:`ExternalGreyBoxModel<pyomo.contrib.pynumero.interfaces.external_grey_box.ExternalGreyBoxModel>`.
We do this by subclassing this base class and implementing methods for function, Jacobian, and, optionally,
Hessian evaluation. The new class will define our external function only; we are not yet connecting
this function to any Pyomo modeling components.

Suppose we would like to represent the function...

.. doctest::
   :skipif: not numpy_available or not scipy_available or not asl_available

    >>> class MyGreyBox:
            ...

Step 2: Construct an ``ExternalGreyBoxBlock``
---------------------------------------------

Step 3: Solve the model with CyIpopt
------------------------------------
