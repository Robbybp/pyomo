import numpy as np
import scipy.sparse as sps

from pyomo.contrib.pynumero.interfaces.nlp import NLP


class VectorValuedExternalFunction(object):
    """ 
    This is a base class for abstract functions. I differentiate this
    from an ExternalGreyBoxModel as I believe the "function" and the
    "model" should be separated, and because I need a different
    Hessian format.

    """

    def n_inputs(self):
        raise NotImplementedError()

    def set_input_values(self, input_values):
        raise NotImplementedError()

    def n_outputs(self):
        raise NotImplementedError()

    def evaluate_outputs(self):
        raise NotImplementedError()

    def evaluate_jacobian_outputs(self):
        # Not all functions have first derivatives
        # TODO: Should I have a separate `DifferentiableFunction` class?
        pass

    def evaluate_hessian_outputs(self):
        # Not all functions have second derivatives
        pass


class IdentityFunction(VectorValuedExternalFunction):
    """
    f: A -> A, f(a) = a
    """

    def __init__(self, n):
        self._dim = n
        self._inputs = np.zeros(n)
        self._outputs = np.zeros(n)
        self._jacobian_outputs = sps.identity(n)
        self._hessian_outputs = [
            sps.coo_matrix((n, n)) for _ in range(n)
        ]

    def n_inputs(self):
        return self._dim

    def n_outputs(self):
        return self._dim

    def set_input_values(self, input_values):
        self._inputs = input_values
        self._outputs = input_values

    def evaluate_outputs(self):
        return self._outputs

    def evaluate_jacobian_outputs(self):
        return self._jacobian_outputs

    def evaluate_hessian_outputs(self):
        return self._hessian_outputs


class FunctionCombination(VectorValuedExternalFunction):

    def __init__(self, *functions):
        """
        Suppose we are given two functions,

        f: A -> B
        g: A -> C

        This class represents h, such that

        h: A -> B x C,
        h(a) = (f(a), g(a))

        This is different from a FunctionStack as all functions use
        the same inputs.

        """
        self._functions = functions
        if len(functions) != 0:
            assert all(f.n_inputs() == functions[0].n_inputs() for f in functions)
            self._n_inputs = functions[0].n_inputs()
        else:
            self._n_inputs = 0
        self._output_partition = self.get_output_partition()

    def get_output_partition(self):
        offset = 0
        output_partition = [None for _ in self._functions]
        for i, f in enumerate(self._functions):
            n_outputs = f.n_outputs()
            start_idx = offset
            end_idx = offset + n_outputs
            output_partition[i] = (start_idx, end_idx)
            offset += n_outputs
        return output_partition

    def get_output_offset(self, i):
        """
        Get the first output coordinate of the i-th function

        Parameters
        ----------
        i: int
            The function coordinate whose output coordinate we want.

        """
        return self._output_partition[i][0]

    def n_inputs(self):
        return self._n_inputs

    def n_outputs(self):
        return sum(f.n_outputs() for f in self._functions)

    def set_input_values(self, input_values):
        for f in functions:
            f.set_input_values(input_values)

    def evaluate_outputs(self):
        outputs = tuple(f.evaluate_outputs() for f in self._functions)
        output = np.concatenate(outputs)
        return output

    def evaluate_jacobian_outputs(self):
        jacobians = tuple(
            f.evaluate_jacobian_outputs() for f in self._functions
            )
        jacobian = sps.vstack(jacobians)
        return jacobian

    def evaluate_hessian_outputs(self):
        hessians = sum(
            (f.evaluate_hessian_outputs() for f in self._functions),
            [],
        )


class FunctionStack(FunctionCombination):

    def __init__(self, *functions):
        """
        Suppose we are given two functions,

        f: A -> C
        g: B -> D

        This class represents h, such that

        h: A x B -> C x D
        h(a, b) = (f(a), g(b))

        """
        self._functions = functions
        self._input_partition = self.get_input_partition()
        self._output_partition = self.get_output_partition()

    def get_input_partition(self):
        offset = 0
        # TODO: We will likely need the mapping from indices to
        # functions to trace coordinates back to their sources
        input_partition = [None for _ in self._functions]
        for i, f in enumerate(self._functions):
            n_inputs = f.n_inputs()
            start_idx = offset
            end_idx = offset + n_inputs
            input_partition[i] = (start_idx, end_idx)
            offset += n_inputs
        return input_partition

    def n_inputs(self):
        return sum(f.n_inputs() for f in self._functions)

    def get_input_offset(self, i):
        """
        Get the first input coordinate of the i-th function

        Parameters
        ----------
        i: int
            The function coordinate whose input coordinate we want.

        """
        return self._input_partition[i][0]

    def set_input_values(self, input_values):
        for (idx1, idx2), f in zip(self._input_partition, self._functions):
            # Assume input_values is compatible with slicing...
            f.set_input_values(input_values[idx1:idx2])

    def evaluate_jacobian_outputs(self):
        jacobians = tuple(
            f.evaluate_jacobian_outputs() for f in self._functions
            )
        jacobian = sps.block_diag(jacobians)
        return jacobian

    def evaluate_hessian_outputs(self):
        # Each Hessian has a single non-zero diagonal block.
        # We replace one of these blocks for each hessian.
        blocks = [
            sps.coo_matrix((f.n_inputs(), f.n_inputs()))
            for f in self._functions
        ]
        hessian = []
        for i, f in enumerate(self._functions):
            cached_block = blocks[i]
            for h in f.evaluate_hessian_outputs():
                blocks[i] = h
                hessian.append(sps.block_diag(blocks))
            blocks[i] = cached_block
        return hessian


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
        outputs2 = self._function2.evaluate_outputs()
        # TODO: permute input values if necessary
        self._function1.set_input_values(outputs2)

    def evaluate_outputs(self):
        return self._function1.evaluate_outputs()

    def evaluate_jacobian_outputs(self):
        """
        Application of the chain rule yields:

        Jfog(x) = Jf(g(x)) * Jg(x)

        Multiplication is standard matrix multiplication.

        """
        jf1 = self._function1.evaluate_jacobian_outputs()
        jf2 = self._function2.evaluate_jacobian_outputs()
        jac = jf1.dot(jf2).tocoo()
        return jac

    def evaluate_hessian_outputs(self):
        """
        Application of the chain rule yields:

        Hfog(x) = (Jf(g(x)) * Hg(x)) + (Jg(x)^T * Hf(g(x)) * Jg(x))

        Mutiplications are matrix-tensor products that return tensors.
        The tensor rank(s) along which each multiply occurs (i.e. which
        "sub-matrix" or "sub-vector" of the tensor gets multiplied)
        are clear from dimensions, and the fact that each matrix along
        the first rank of the resulting tensor must be symmetric.

        The two terms in the above right-hand-side are referred to
        as term1 and term2 in the code.

        """
        # In my applications so far, Jacobian evaluation
        # has been very fast. If this changes, I may
        # have to cache these matrices.
        jf1 = self._function1.evaluate_jacobian_outputs()
        jf2 = self._function2.evaluate_jacobian_outputs()

        # TODO: I can improve performance by only using the
        # lower triangle of Hessian matrices.
        hf1 = self._function1.evaluate_hessian_outputs()
        hf2 = self._function2.evaluate_hessian_outputs()

        n_in = self._function2.n_inputs()
        n_f2_out = self._function2.n_outputs()

        # Multiply jf2 by each matrix defined by a coordinate of the
        # first rank of hf1
        term2 = [jf2.transpose().dot(H).dot(jf2) for H in hf1]

        # Now need to multiply jf1 by each of nx^2 vectors

        # Get rows of "flattened tensor" matrix
        hf2_flat_rows = [H.reshape((1, n_in**2)).tocsr() for H in hf2]
        # Unclear whether we should convert these matrices to CSR.
        # SciPy claims this will be faster. Seems true for n_in > ~5000.
        # TODO: Benchmark in real problems.

        # Stack rows
        hf2_flat = sps.vstack(hf2_flat_rows)

        # Multiply flattened Hessian-2 by Jacobian-1
        term1_flat = jf1.dot(hf2_flat).tocsr()
        # Very important to make sure this matrix is in CSR format for
        # fast getrow below.

        # Each row of the matrix becomes a matrix of the tensor
        term1 = [
            term1_flat.getrow(i).reshape((n_in, n_in)) for i in range(n_f2_out)
            ]
        hessian = [(mat1 + mat2).tocoo() for mat1, mat2 in zip(term1, term2)]

        return hessian


class FunctionFromNLP(VectorValuedExternalFunction):

    def __init__(self, nlp):
        """
        Options to support, eventually:
        - Include vs. not include the objective
        - include a subset of the constraint bodies as outputs
        - include a subset of the variables as inputs
        """
        self._nlp = nlp
        self._include_objective = True
        self._input_coords = np.arange(self._nlp.n_primals())
        self._output_coords = np.arange(self._nlp.n_constraints())

    def n_inputs(self):
        return len(self._input_coords)

    def n_outputs(self):
        return int(self._include_objective) + len(self._output_coords)

    def set_input_values(self, input_values):
        self._nlp.set_primals(input_values)

    def evaluate_outputs(self):
        # TODO: Extract subvector
        constraints = self._nlp.evaluate_constraints()
        # Here we assume the objects returned by the nlp
        # evaluation routines will by compatible with NumPy.
        if self._include_objective:
            objective = np.array([self._nlp.evaluate_objective()])
            return np.concatenate((objective, constraints))
        else:
            return constraints

    def evaluate_jacobian_outputs(self):
        # TODO: Extract submatrix
        con_jac = self._nlp.evaluate_jacobian()
        if self._include_objective:
            obj_grad = self._nlp.evaluate_grad_objective()
            obj_grad = obj_grad.reshape((1, self.n_inputs()))
            obj_grad = sps.coo_matrix(obj_grad)
            # Here we assume something about the "orientation"
            # of the Jacobian.
            return sps.vstack((obj_grad, con_jac))
        else:
            return con_jac

    def evaluate_hessian_outputs(self):
        con_offset = int(self._include_objective)
        out_hess = [None for _ in range(self.n_outputs())]

        cached_duals = self._nlp.get_duals()
        cached_obj_factor = self._nlp.get_obj_factor()
        self._nlp.set_obj_factor(0.0)

        duals = np.zeros(self._nlp.n_constraints())
        self._nlp.set_duals(duals)

        if self._include_objective:
            self._nlp.set_obj_factor(1.0)
            out_hess[0] = self._nlp.evaluate_hessian_lag()
            self._nlp.set_obj_factor(0.0)

        for i in range(con_offset, self.n_outputs()):
            # TODO: outputs don't necessarily include all constraints in order
            con_idx = i - con_offset
            duals[con_idx] = 1.0
            self._nlp.set_duals(duals)
            # TODO: restrict Hessian to variables that are inputs
            out_hess[i] = self._nlp.evaluate_hessian_lag()
            duals[con_idx] = 0.0

        self._nlp.set_duals(cached_duals)
        self._nlp.set_obj_factor(cached_obj_factor)

        return out_hess


class NLPFromFunction(NLP):

    def __init__(self, function):
        self._function = function

        self._primals = np.zeros(function.n_inputs())

        # TODO: Flag for whether objective is included in the function
        self._n_constraints = function.n_outputs() - 1

    def n_primals(self):
        return self._function.n_inputs()

    def n_constraints(self):
        return self._n_constraints
