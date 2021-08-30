import numpy as np
import scipy.sparse as sps

from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.interfaces.utils import CondensedSparseSummation


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

    def get_input_source(self, idx):
        assert idx >= 0 and idx < self.n_inputs()
        return self, idx

    def get_output_source(self, idx):
        assert idx >= 0 and idx < self.n_outputs()
        return self, idx


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
            assert all(
                f.n_inputs() == functions[0].n_inputs() for f in functions
            )
            self._n_inputs = functions[0].n_inputs()
        else:
            self._n_inputs = 0
        self._output_partition = self.get_output_partition()
        self._output_sources = [
            i for i, (start_idx, end_idx) in enumerate(self._output_partition)
            for _ in range(start_idx, end_idx)
        ]

    def get_input_source(self, idx):
        # What to do here... This is not well-defined
        assert idx >= 0 and idx < self.n_inputs()
        return self, idx

    def get_output_source(self, idx):
        fcn_idx = self._output_sources[idx]
        offset = self._output_partition[fcn_idx][0]
        return self._functions[fcn_idx].get_output_source(idx - offset)

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
        for f in self._functions:
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
        return hessians


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
        self._input_sources = [
            i for i, (start_idx, end_idx) in enumerate(self._input_partition)
            for _ in range(start_idx, end_idx)
        ]
        self._output_partition = self.get_output_partition()
        self._output_sources = [
            i for i, (start_idx, end_idx) in enumerate(self._output_partition)
            for _ in range(start_idx, end_idx)
        ]

    def get_input_source(self, idx):
        fcn_idx = self._input_sources[idx]
        offset = self._input_partition[fcn_idx][0]
        return self._functions[fcn_idx].get_input_source(idx - offset)

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

    def get_input_source(self, idx):
        return self._function2.get_input_source(idx)

    def get_output_source(self, idx):
        return self._function1.get_output_source(idx)

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

    def __init__(self, nlp, include_objective=True):
        """
        Options to support, eventually:
        - Include vs. not include the objective
        - include a subset of the constraint bodies as outputs
        - include a subset of the variables as inputs
        """
        self._nlp = nlp
        self._include_objective = include_objective
        self._input_coords = np.arange(self._nlp.n_primals())
        self._output_coords = np.arange(self._nlp.n_constraints())

    def get_input_source(self, idx):
        """
        idx is a variable/column index in the NLP
        """
        assert idx >= 0 and idx < self.n_inputs()
        # TODO: Get the nlp index corresponding to this function
        # index. For now they are the same.
        return self._nlp, idx

    def get_output_source(self, idx):
        """
        idx is a constraint/objective/row index in the NLP
        """
        # Need a way to encode whether idx is an objective
        # For now, -1 corresponds to the objective...
        assert idx >= 0 and idx < self.n_outputs()
        return self._nlp, idx - int(self._include_objective)

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

    def __init__(
            self,
            function,
            primals_lb=None,
            primals_ub=None,
            constraints_lb=None,
            constraints_ub=None,
            ):
        primals_lb = {} if primals_lb is None else primals_lb
        primals_ub = {} if primals_ub is None else primals_ub
        constraints_lb = {} if constraints_lb is None else constraints_lb
        constraints_ub = {} if constraints_ub is None else constraints_ub
        self._function = function

        # TODO: Flag for whether objective is included in the function
        self._objective_included = True
        con_offset = int(self._objective_included)
        self._n_constraints = function.n_outputs() - con_offset

        self._primals = self.init_primals()
        self._duals = self.init_duals()
        self._obj_factor = 1.0

        # Setup bound data structures
        self._primals_lb = np.array([-np.inf for _ in range(self.n_primals())])
        self._primals_ub = np.array([np.inf for _ in range(self.n_primals())])
        self._constraints_lb = np.array(
            [-np.inf for _ in range(self.n_constraints())]
        )
        self._constraints_ub = np.array(
            [np.inf for _ in range(self.n_constraints())]
        )

        # Assuming these are dicts for now
        assert isinstance(primals_lb, dict)
        assert isinstance(primals_ub, dict)
        assert isinstance(constraints_lb, dict)
        assert isinstance(constraints_ub, dict)

        # TODO: Should be able to do this with vectorized syntax
        for i in range(self.n_primals()):
            if i in primals_lb:
                self._primals_lb[i] = primals_lb[i]
            else:
                obj, source_idx = self._function.get_input_source(i)
                if isinstance(obj, NLP):
                    self._primals_lb[i] = obj.primals_lb()[source_idx]
            if i in primals_ub:
                self._primals_ub[i] = primals_ub[i]
            else:
                obj, source_idx = self._function.get_input_source(i)
                if isinstance(obj, NLP):
                    self._primals_ub[i] = obj.primals_ub()[source_idx]

        for i in range(self.n_constraints()):
            if i in constraints_lb:
                self._constraints_lb[i] = constraints_lb[i]
            else:
                obj, src_idx = self._function.get_output_source(i + con_offset)
                # src_idx should be a valid index into the NLP's constraints.
                # This is how get_output_source is handled by FunctionFromNLP
                if isinstance(obj, NLP):
                    self._constraints_lb[i] = obj.constraints_lb()[src_idx]
            if i in constraints_ub:
                self._constraints_ub[i] = constraints_ub[i]
            else:
                obj, src_idx = self._function.get_output_source(i + con_offset)
                if isinstance(obj, NLP):
                    self._constraints_ub[i] = obj.constraints_ub()[src_idx]

        self._hessian_sum = None

    def n_primals(self):
        return self._function.n_inputs()

    def n_constraints(self):
        return self._n_constraints

    def nnz_jacobian(self):
        """
        This is nontrivial to do in general without just getting the numeric
        Jacobian. CyIpoptNLP does this anyway, so no need to repeat the code
        here for now.

        """
        raise NotImplementedError()

    def nnz_hessian_lag(self):
        """
        This is nontrivial to do in general without just getting the numeric
        Hessian, and is not necessary for our immediate CyIpopt application.

        """
        raise NotImplementedError()

    def primals_lb(self):
        return np.copy(self._primals_lb)

    def primals_ub(self):
        return np.copy(self._primals_ub)

    def constraints_lb(self):
        return np.copy(self._constraints_lb)

    def constraints_ub(self):
        return np.copy(self._constraints_ub)

    def init_primals(self):
        return np.zeros(self.n_primals())

    def init_duals(self):
        return np.zeros(self.n_constraints())

    def create_new_vector(self):
        # I do not know where this is necessary.
        raise NotImplementedError()

    def set_primals(self, primals):
        np.copyto(self._primals, primals)
        self._function.set_input_values(primals)

    def get_primals(self):
        return self._primals.copy()

    def set_duals(self, duals):
        np.copyto(self._duals, duals)

    def get_duals(self):
        return self._duals.copy()

    def set_obj_factor(self, obj_factor):
        self._obj_factor = obj_factor

    def get_obj_factor(self):
        return self._obj_factor

    def get_obj_scaling(self):
        return 1.0

    def get_primals_scaling(self):
        return np.ones(self.n_primals())

    def get_constraints_scaling(self):
        return np.ones(self.n_constraints())

    def evaluate_objective(self):
        if self._objective_included:
            return self._function.evaluate_outputs()[0]
        else:
            # TODO: option for user-provided objective?
            return 0.0

    def evaluate_grad_objective(self, out=None):
        if self._objective_included:
            coo = self._function.evaluate_jacobian_outputs()
            # TODO: cache these matrices
            csr = coo.tocsr()
            return csr[0, :].toarray()[0]
        else:
            return np.zeros(self.n_primals())

    def evaluate_constraints(self, out=None):
        outputs = self._function.evaluate_outputs()
        if self._objective_included:
            return np.copy(outputs[1:])
        else:
            return np.copy(outputs)

    def evaluate_jacobian(self, out=None):
        coo = self._function.evaluate_jacobian_outputs()
        if self._objective_included:
            csr = coo.tocsr()
            return csr[1:, :].tocoo()
        else:
            return coo

    def evaluate_hessian_lag(self, out=None):
        hessian = self._function.evaluate_hessian_outputs()
        if self._hessian_sum is None:
            self._hessian_sum = CondensedSparseSummation(hessian)
        if self._objective_included:
            obj_factor_array = np.array([self.get_obj_factor()])
        else:
            obj_factor_array = np.array([])
        # NOTE: assuming here that if objective is not provided by the
        # function, it has no contibution to the Hessian.
        to_multiply = np.concatenate((obj_factor_array, self.get_duals()))
        sum_ = self._hessian_sum.sum(
            list(mult*hess for mult, hess in zip(to_multiply, hessian))
        )
        return sum_

    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError()
