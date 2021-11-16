from pyomo.contrib.pynumero.interfaces.nlp import NLP
import numpy as np
import scipy.sparse as sp

class _BaseNLPDelegator(NLP):
    def __init__(self, original_nlp):
        """
        This is a base class to make it easier to implement NLP
        classes that wrap other NLP instances. This class
        simply reproduces the NLP interface by passing the call
        onto the original_nlp passed in the constructor. This allows
        new wrapper classes to only implement the methods that change,
        allowing the others to pass through.

        Parameters
        ----------
        original_nlp : NLP-like
            The original NLP object that we want to wrap
        """
        super(NLP, self).__init__()
        self._original_nlp = original_nlp

    def n_primals(self):
        return self._original_nlp.n_primals()

    def primals_names(self):
        return self._original_nlp.primals_names()

    def n_constraints(self):
        return self._original_nlp.n_constraints()

    def constraint_names(self):
        return self._original_nlp.constraint_names()

    def nnz_jacobian(self):
        return self._original_nlp.nnz_jacobian()

    def nnz_hessian_lag(self):
        return self._original_nlp.nnz_hessian_lag()

    def primals_lb(self):
        return self._original_nlp.primals_lb()

    def primals_ub(self):
        return self._original_nlp.primals_ub()

    def constraints_lb(self):
        return self._original_nlp.constraints_lb()

    def constraints_ub(self):
        return self._original_nlp.constraints_ub()

    def init_primals(self):
        return self._original_nlp.init_primals()

    def init_duals(self):
        return self._original_nlp.init_duals()

    def create_new_vector(self, vector_type):
        return self._original_nlp.create_new_vector(vector_type)

    def set_primals(self, primals):
        self._original_nlp.set_primals(primals)

    def get_primals(self):
        return self._original_nlp.get_primals()

    def set_duals(self, duals):
        self._original_nlp.set_duals(duals)

    def get_duals(self):
        return self._original_nlp.get_duals()

    def set_obj_factor(self, obj_factor):
        self._original_nlp.set_obj_factor(obj_factor)

    def get_obj_factor(self):
        return self._original_nlp.get_obj_factor()

    def get_obj_scaling(self):
        return self._original_nlp.get_obj_scaling()

    def get_primals_scaling(self):
        return self._original_nlp.get_primals_scaling()

    def get_constraints_scaling(self):
        return self._original_nlp.get_constraints_scaling()

    def evaluate_objective(self):
        return self._original_nlp.evaluate_objective()

    def evaluate_grad_objective(self, out=None):
        return self._original_nlp.evaluate_grad_objective(out)

    def evaluate_constraints(self, out=None):
        return self._original_nlp.evaluate_constraints(out)

    def evaluate_jacobian(self, out=None):
        return self._original_nlp.evaluate_jacobian(out)

    def evaluate_hessian_lag(self, out=None):
        return self._original_nlp.evaluate_hessian_lag(out)

    def report_solver_status(self, status_code, status_message):
        self._original_nlp.report_solver_status(status_code, status_message)


class RenamedNLP(_BaseNLPDelegator):
    def __init__(self, original_nlp, primals_name_map):
        """
        This class takes an NLP that and allows one to rename the primals.
        It is a thin wrapper around the original NLP.

        Parameters
        ----------
        original_nlp : NLP-like
            The original NLP object that implements the NLP interface

        primals_name_map : dict of str --> str
            This is a dictionary that maps from the names
            in the original NLP class to the desired names
            for this instance.
        """
        super(RenamedNLP, self).__init__(original_nlp)
        self._primals_name_map = primals_name_map
        self._new_primals_names = None
        # Todo: maybe do this on first call instead of __init__?
        self._generate_new_names()

    def _generate_new_names(self):
        if self._new_primals_names is None:
            assert self._original_nlp.n_primals() == len(self._primals_name_map)
            self._new_primals_names = \
                [self._primals_name_map[nm] for nm in self._original_nlp.primals_names()]
            
    def primals_names(self):
        return self._new_primals_names


class ProjectedNLP(_BaseNLPDelegator):
    def __init__(
            self,
            original_nlp,
            primals_ordering,
            constraints_ordering=None,
            include_objective=True,
            ):
        """
        This class takes an NLP that depends on a set of primals (original
        space) and converts it to an NLP that depends on a reordered set of 
        primals (projected space).

        This will impact all the returned items associated with primal
        variables. E.g., the gradient will be in the new primals ordering
        instead of the original primals ordering.

        Note also that this can include additional primal variables not
        in the original NLP, or can exclude primal variables that were 
        in the original NLP.

        Parameters
        ----------
        original_nlp : NLP-like
            The original NLP object that implements the NLP interface

        primals_ordering: list
            List of strings indicating the desired primal variable 
            ordering for this NLP. The list can contain new variables
            that are not in the original NLP, thereby expanding the 
            space of the primal variables.

        constraints_ordering: list
            List of coordinates indicating the desired constraints
            in the original NLP to appear in this NLP.
        """
        super(ProjectedNLP, self).__init__(original_nlp)
        self._primals_ordering = list(primals_ordering)
        self._original_idxs = None
        self._projected_idxs = None
        self._generate_maps()
        self._projected_primals = self.init_primals()
        self._jacobian_nz_mask = None
        self._hessian_nz_mask = None
        self._nnz_jacobian = None
        self._nnz_hessian_lag = None

        original_con_coords = np.arange(self._original_nlp.n_constraints())
        if constraints_ordering is None:
            constraints_ordering = original_con_coords
        # NOTE: constraints_ordering is a list (or array) of coordinates,
        # and should only contain coordinates valid for this NLP...
        constraints_ordering = np.array(constraints_ordering)
        if not np.all(constraints_ordering <= original_nlp.n_constraints()):
            raise ValueError(
                "Constraint coordinates must be valid for the original NLP"
            )
        self._constraints_ordering = constraints_ordering
        mask = ~np.isin(original_con_coords, constraints_ordering)
        self._other_constraint_coords = original_con_coords[mask]
        self._projected_constraint_map = {
            j: i for i, j in enumerate(self._constraints_ordering)
        }
        self._include_objective = include_objective

    def _generate_maps(self):
        if self._original_idxs is None or self._projected_idxs is None:
            primals_ordering_dict = {k:i for i,k in enumerate(self._primals_ordering)}
            original_names = self._original_nlp.primals_names()
            original_idxs = list()
            projected_idxs = list()
            for i,nm in enumerate(original_names):
                if nm in primals_ordering_dict:
                    # we need the reordering for this element
                    original_idxs.append(i)
                    projected_idxs.append(primals_ordering_dict[nm])
            self._original_idxs = np.asarray(original_idxs)
            self._projected_idxs = np.asarray(projected_idxs)
            self._original_to_projected = np.nan*np.zeros(self._original_nlp.n_primals())
            self._original_to_projected[self._original_idxs] = self._projected_idxs

    def n_primals(self):
        return len(self._primals_ordering)

    def primals_names(self):
        return list(self._primals_ordering)

    def nnz_jacobian(self):
        if self._nnz_jacobian is None:
            J = self.evaluate_jacobian()
            self._nnz_jacobian = len(J.data)
        return self._nnz_jacobian

    def nnz_hessian_lag(self):
        if self._nnz_hessian_lag is None:
            H = self.evaluate_hessian_lag()
            self._nnz_hessian_lag = len(H.data)
        return self._nnz_hessian_lag

    def _project_primals(self, default, original_primals):
        projected_x = default*np.ones(self.n_primals(), dtype=np.float64)
        projected_x[self._projected_idxs] = original_primals[self._original_idxs]
        return projected_x

    def primals_lb(self):
        return self._project_primals(-np.inf, self._original_nlp.primals_lb())

    def primals_ub(self):
        return self._project_primals(np.inf, self._original_nlp.primals_ub())

    def init_primals(self):
        # Todo: think about what to do here if an entry is not defined
        # for now, we default to NaN, but there may be a better way
        # (e.g., taking a new initial value in the constructor?)
        return self._project_primals(np.nan, self._original_nlp.init_primals())

    def create_new_vector(self, vector_type):
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        return self._original_nlp.create_new_vector(vector_type)

    def set_primals(self, primals):
        # here, we keep a local copy of the projected primals so
        # we can give back these values in get_primals. This might
        # not be the best idea since we can't really support this
        # same strategy for other methods (e.g., init_primals) where
        # we now use NaNs to fill in any "missing" entries.
        np.copyto(self._projected_primals, primals)
        original_primals = self._original_nlp.get_primals()
        original_primals[self._original_idxs] = primals[self._projected_idxs]
        self._original_nlp.set_primals(original_primals)

    def get_primals(self):
        original_primals = self._original_nlp.get_primals()
        self._projected_primals[self._projected_idxs] = original_primals[self._original_idxs]
        return self._projected_primals

    def get_primals_scaling(self):
        return self._project_primals(np.nan, self._original_nlp.get_primals_scaling())

    def evaluate_grad_objective(self, out=None):
        original_grad_objective = self._original_nlp.evaluate_grad_objective()
        projected_objective = self._project_primals(0.0, original_grad_objective) 
        if out is None:
            return projected_objective
        np.copyto(out, projected_objective)
        return out

    def n_constraints(self):
        return len(self._constraints_ordering)

    def evaluate_constraints(self, out=None):
        original_constraints = self._original_nlp.evaluate_constraints()
        con_order = self._constraints_ordering
        self._projected_constraints = original_constraints[con_order]
        if out is None:
            return self._projected_constraints
        np.copyto(out, self._projected_constraints)
        return out

    def evaluate_jacobian(self, out=None):
        original_jacobian = self._original_nlp.evaluate_jacobian()
        if out is not None:
            np.copyto(out.data, original_jacobian.data[self._jacobian_nz_mask])
            return out

        row = original_jacobian.row
        col = original_jacobian.col
        data = original_jacobian.data

        if self._jacobian_nz_mask is None:
            # need to remap the irow, jcol to the new space and change the size
            col_nz_mask = np.isin(col, self._original_idxs)
            # row_nz_mask identifies which nonzeros belong to a row
            # (constraint) that is retained by the projected NLP
            row_nz_mask = np.isin(row, self._constraints_ordering)
            self._jacobian_nz_mask = col_nz_mask & row_nz_mask

        new_col = col[self._jacobian_nz_mask]
        new_col = self._original_to_projected[new_col]
        new_row = row[self._jacobian_nz_mask]
        new_row = np.fromiter(
            (self._projected_constraint_map[r] for r in new_row), dtype=int
        )
        new_data = data[self._jacobian_nz_mask]

        return sp.coo_matrix(
            (new_data, (new_row,new_col)),
            shape=(self.n_constraints(), self.n_primals()),
        )

    def init_duals(self):
        orig_duals = self._original_nlp.init_duals()
        duals = orig_duals[self._constraints_ordering]
        return duals

    def get_duals(self):
        orig_duals = self._original_nlp.get_duals()
        proj_duals = orig_duals[self._constraints_ordering]
        return proj_duals

    def set_duals(self, duals):
        orig_duals = self._original_nlp.get_duals()
        orig_duals[self._constraints_ordering] = duals
        self._original_nlp.set_duals(orig_duals)

    def evaluate_hessian_lag(self, out=None):
        # Cache original NLP's duals and temporarily set those for
        # constraints we don't want to zero.
        cached_obj_factor = self._original_nlp.get_obj_factor()
        if not self._include_objective:
            self._original_nlp.set_obj_factor(0.0)
        cached_duals = self._original_nlp.get_duals()
        duals = np.copy(cached_duals)
        n_other_constraints = len(self._other_constraint_coords)
        duals[self._other_constraint_coords] = np.zeros(n_other_constraints)
        self._original_nlp.set_duals(duals)

        original_hessian = self._original_nlp.evaluate_hessian_lag()
        if out is not None:
            np.copyto(out.data, original_hessian.data[self._hessian_nz_mask])
            return out

        row = original_hessian.row
        col = original_hessian.col
        data = original_hessian.data

        if self._hessian_nz_mask is None:
            # need to remap the irow, jcol to the new space and change the size
            self._hessian_nz_mask = np.isin(col, self._original_idxs) & np.isin(row, self._original_idxs)

        new_col = col[self._hessian_nz_mask]
        new_col = self._original_to_projected[new_col]
        new_row = row[self._hessian_nz_mask]
        new_row = self._original_to_projected[new_row]
        new_data = data[self._hessian_nz_mask]

        # Reset duals and objective factor of original NLP
        if not self._include_objective:
            self._original_nlp.set_obj_factor(cached_obj_factor)
        self._original_nlp.set_duals(cached_duals)

        return sp.coo_matrix((new_data, (new_row,new_col)), shape=(self.n_primals(), self.n_primals()))

    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError('Need to think about this...')
