#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.common.dependencies import attempt_import
from pyomo.common.timing import HierarchicalTimer

mpi_block_vector, mpi_block_vector_available = attempt_import(
    'pyomo.contrib.pynumero.sparse.mpi_block_vector'
)


def build_bounds_mask(vector):
    """
    Creates masks for converting from the full vector of bounds that
    may contain -np.inf or np.inf to a vector of bounds that are finite
    only.
    """
    return build_compression_mask_for_finite_values(vector)


def build_compression_matrix(compression_mask):
    """
    Return a sparse matrix CM of ones such that
    compressed_vector = CM*full_vector based on the
    compression mask

    Parameters
    ----------
    compression_mask: np.ndarray or pyomo.contrib.pynumero.sparse.block_vector.BlockVector

    Returns
    -------
    cm: coo_matrix or BlockMatrix
       The compression matrix
    """
    if isinstance(compression_mask, BlockVector):
        n = compression_mask.nblocks
        res = BlockMatrix(nbrows=n, nbcols=n)
        for ndx, block in enumerate(compression_mask):
            sub_matrix = build_compression_matrix(block)
            res.set_block(ndx, ndx, sub_matrix)
        return res
    elif type(compression_mask) is np.ndarray:
        cols = compression_mask.nonzero()[0]
        nnz = len(cols)
        rows = np.arange(nnz, dtype=np.int64)
        data = np.ones(nnz)
        return coo_matrix((data, (rows, cols)), shape=(nnz, len(compression_mask)))
    elif isinstance(compression_mask, mpi_block_vector.MPIBlockVector):
        from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix

        n = compression_mask.nblocks
        rank_ownership = np.ones((n, n), dtype=np.int64) * -1
        for i in range(n):
            rank_ownership[i, i] = compression_mask.rank_ownership[i]
        res = MPIBlockMatrix(
            nbrows=n,
            nbcols=n,
            rank_ownership=rank_ownership,
            mpi_comm=compression_mask.mpi_comm,
            assert_correct_owners=False,
        )
        for ndx in compression_mask.owned_blocks:
            block = compression_mask.get_block(ndx)
            sub_matrix = build_compression_matrix(block)
            res.set_block(ndx, ndx, sub_matrix)
        return res


def build_compression_mask_for_finite_values(vector):
    """
    Creates masks for converting from the full vector of
    values to the vector that contains only the finite values. This is
    typically used to convert a vector of bounds (that may contain np.inf
    and -np.inf) to only the bounds that are finite.
    """
    full_finite_mask = np.isfinite(vector)
    return full_finite_mask


# TODO: Is this needed anywhere?
# def build_expansion_map_for_finite_values(vector):
#    """
#    Creates a map from the compressed vector to the full
#    vector based on the locations of finite values only. This is
#    typically used to map a vector of bounds (that is compressed to only
#    contain the finite values) to a full vector (that may contain np.inf
#    and -np.inf).
#    """
#    full_finite_mask = np.isfinite(vector)
#    finite_full_map = full_finite_mask.nonzero()[0]
#    return finite_full_map


def full_to_compressed(full_array, compression_mask, out=None):
    if out is not None:
        np.compress(compression_mask, full_array, out=out)
        return out
    else:
        return np.compress(compression_mask, full_array)


def compressed_to_full(compressed_array, compression_mask, out=None, default=None):
    if out is None:
        ret = np.empty(len(compression_mask))
        ret.fill(np.nan)
    else:
        ret = out

    ret[compression_mask] = compressed_array
    if default is not None:
        ret[~compression_mask] = default

    return ret


def make_lower_triangular_full(lower_triangular_matrix):
    '''
    This function takes a symmetric matrix that only has entries in the
    lower triangle and makes is a full matrix by duplicating the entries
    '''
    mask = lower_triangular_matrix.row != lower_triangular_matrix.col

    row = np.concatenate(
        (lower_triangular_matrix.row, lower_triangular_matrix.col[mask])
    )
    col = np.concatenate(
        (lower_triangular_matrix.col, lower_triangular_matrix.row[mask])
    )
    data = np.concatenate(
        (lower_triangular_matrix.data, lower_triangular_matrix.data[mask])
    )

    return coo_matrix((data, (row, col)), shape=lower_triangular_matrix.shape)


class CondensedSparseSummation(object):
    def __init__(self, list_of_matrices):
        """
        This class is used to perform a summation of sparse matrices
        while retaining the correct and consistent nonzero structure.
        Create the class with the list of matrices you want to sum,
        and the condensed_summation method remains valid as long as
        the structure of the individual matrices is consistent
        """
        self._nz_tuples = None
        self._maps = None
        self._build_maps(list_of_matrices)

    def _build_maps(self, list_of_matrices):
        """
        This method creates the maps that are used in condensed_sum.
        These maps remain valid as long as the nonzero structure of
        the individual matrices does not change
        """
        # get the list of all unique nonzeros across the matrices
        nz_tuples = set()
        for m in list_of_matrices:
            nz_tuples.update(zip(m.row, m.col))
        nz_tuples = sorted(nz_tuples)
        self._nz_tuples = nz_tuples
        if self._nz_tuples:
            # This line fails if there are no nonzeros
            self._row, self._col = list(zip(*nz_tuples))
        else:
            self._row = np.array([], dtype=int)
            self._col = np.array([], dtype=int)
        row_col_to_nz_map = {t: i for i, t in enumerate(nz_tuples)}

        self._shape = None
        self._maps = list()
        for m in list_of_matrices:
            nnz = len(m.data)
            map_row = np.zeros(nnz)
            map_col = np.zeros(nnz)
            for i in range(nnz):
                map_col[i] = i
                map_row[i] = row_col_to_nz_map[(m.row[i], m.col[i])]
            mp = coo_matrix(
                (np.ones(nnz), (map_row, map_col)), shape=(len(row_col_to_nz_map), nnz)
            )
            self._maps.append(mp)
            if self._shape is None:
                self._shape = m.shape
            else:
                assert self._shape == m.shape

    def sum(self, list_of_matrices):
        data = np.zeros(len(self._row))
        assert len(self._maps) == len(list_of_matrices)
        for i, mp in enumerate(self._maps):
            data += mp.dot(list_of_matrices[i].data)
        ret = coo_matrix(
            (data, (np.copy(self._row), np.copy(self._col))), shape=self._shape
        )
        return ret


def structure_preserving_product(mat1, mat2):
    # TODO: Performance considerations of converting to COO?
    mat1 = mat1.tocoo()
    mat2 = mat2.tocoo()

    mat1_ones = coo_matrix(
        (np.ones(mat1.nnz), (mat1.row, mat1.col)), shape=mat1.shape, dtype=int
    )
    mat2_ones = coo_matrix(
        (np.ones(mat2.nnz), (mat2.row, mat2.col)), shape=mat2.shape, dtype=int
    )
    structural_prod = mat1_ones.dot(mat2_ones).tocoo()
    numeric_prod = mat1.dot(mat2).tocoo()

    assert structural_prod.shape == numeric_prod.shape
    nrow, ncol = structural_prod.shape

    # Get the coordinates in the row-major "flattened array" of the nonzeros
    structural_flat_coords = ncol*structural_prod.row + structural_prod.col
    numeric_flat_coords = ncol*numeric_prod.row + numeric_prod.col

    # Get elements that are in the structural nonzeros but not numeric nonzeros
    mask = ~np.isin(structural_flat_coords, numeric_flat_coords)
    explicit_zero_flat_coords = structural_flat_coords[mask]

    explicit_zero_row, explicit_zero_col = np.divmod(explicit_zero_flat_coords, ncol)
    n_explicit_zero = explicit_zero_flat_coords.shape[0]
    explicit_zero_data = np.zeros(n_explicit_zero, dtype=float)

    prod_row = np.concatenate((numeric_prod.row, explicit_zero_row))
    prod_col = np.concatenate((numeric_prod.col, explicit_zero_col))
    prod_data = np.concatenate((numeric_prod.data, explicit_zero_data))
    prod = coo_matrix((prod_data, (prod_row, prod_col)), shape=(nrow, ncol))
    return prod


import numpy as np
import scipy.sparse as sps
import itertools
from pyomo.contrib.incidence_analysis.triangularize import (
    _get_scc_dag_of_projection,
)
import networkx.algorithms.bipartite as nxb
import networkx.algorithms.dag as nxd


def _dense_to_full_sparse(matrix):
    """
    Used to convert a dense matrix (2d NumPy array) to SciPy sparse matrix
    with explicit coordinates for every entry, including zeros. This is
    used because _ExternalGreyBoxAsNLP methods rely on receiving sparse
    matrices where sparsity structure does not change between calls.
    This is difficult to achieve for matrices obtained via the implicit
    function theorem unless an entry is returned for every coordinate
    of the matrix.

    Note that this does not mean that the Hessian of the entire NLP will
    be dense, only that the block corresponding to this external model
    will be dense.
    """
    # TODO: Allow methods to hard-code Jacobian/Hessian sparsity structure
    # in the case it is known a priori.
    # TODO: Decompose matrices to infer maximum fill-in sparsity structure.
    nrow, ncol = matrix.shape
    row = []
    col = []
    data = []
    for i, j in itertools.product(range(nrow), range(ncol)):
        row.append(i)
        col.append(j)
        data.append(matrix[i, j])
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return sps.coo_matrix((data, (row, col)), shape=(nrow, ncol))


def block_triangularize_with_dag(matrix):
    nrow, ncol = matrix.shape
    graph = nxb.matrix.from_biadjacency_matrix(matrix)
    row_nodes = list(range(nrow))
    matching = nxb.maximum_matching(graph, top_nodes=row_nodes)
    scc_list, dag = _get_scc_dag_of_projection(graph, row_nodes, matching)
    # This order maps coordinates in the sorted space to coordinates in the
    # original space.
    scc_order = list(nxd.lexicographical_topological_sort(dag))
    sccs = [
        sorted([(i, matching[i]) for i in scc_list[scc_idx]])
        for scc_idx in scc_order
    ]
    row_partition = [[i for i, j in scc] for scc in sccs]
    col_partition = [[j - nrow for i, j in scc] for scc in sccs]

    original_to_sorted = np.argsort(scc_order)

    # Reverse so that we map each SCC to those it depends on.
    rev_dag = dag.reverse()
    # Store DAG as adjacency list
    dag_ll = [list(rev_dag[i]) for i in range(len(sccs))]

    # Map adjacent nodes to nodes in the sorted space
    # This is like looking up the "location" of each node.
    dag_ll = [[original_to_sorted[idx] for idx in adj] for adj in dag_ll]

    # These nodes are implicitly already in sorted order
    # This is like putting the nodes in order
    dag_ll = [dag_ll[idx] for idx in scc_order]
    return row_partition, col_partition, dag_ll


def extract_submatrix(matrix, rows, cols):
    """
    Parameters
    ----------
    matrix: scipy.sparse matrix
    rows: numpy.ndarray
        Array of row indices
    cols: numpy.ndarray
        Array of col indices

    """
    matrix = matrix.tocoo()
    nrow, ncol = matrix.shape
    sub_nrow = len(rows)
    sub_ncol = len(cols)

    # Create mask indicating which matrix entries to keep
    mask = np.isin(matrix.row, rows)
    mask &= np.isin(matrix.col, cols)
    # In addition to knowing whether we are in "rows", I would like to
    # know where we are in "rows"

    # Create maps from original coordinates to their locations in submatrix
    # TODO: What is a good placeholder here?
    row_old_to_new = -np.ones(nrow)
    # Replace coordinates we want to extract with their location in the
    # provided coordinate arrays.
    row_old_to_new[rows] = np.arange(sub_nrow)
    col_old_to_new = -np.ones(ncol)
    col_old_to_new[cols] = np.arange(sub_ncol)

    # Extract only the entries we want to keep
    sub_row = matrix.row[mask]
    sub_col = matrix.col[mask]
    sub_data = matrix.data[mask]
    # I want the location of each of these coordinates in the user-provided
    # coordinate arrays
    # Seems like there should be a more efficient way to do this.

    # Map coordinates in full matrix to coordinates in submatrix
    sub_row = row_old_to_new[sub_row]
    sub_col = col_old_to_new[sub_col]

    # Reorder 
    submatrix = sps.coo_matrix(
        (sub_data, (sub_row, sub_col)), shape=(sub_nrow, sub_ncol)
    )
    return submatrix


def structure_preserving_solve(matrix, rhs):
    """
    Parameters
    ----------
    matrix: scipy.sparse matrix
        The square matrix defining the linear system to solve
    rhs: scipy.sparse matrix
        The right-hand-side matrix

    Returns
    -------
    scipy.sparse matrix
        The solution matrix

    """
    nrow, ncol = matrix.shape
    assert nrow == ncol
    dim = nrow
    nrow, nrhs = rhs.shape
    assert nrow == dim

    rhs_coords = np.arange(nrhs)

    rblocks, cblocks, dag = block_triangularize_with_dag(matrix)
    lhs_submatrices = [
        extract_submatrix(matrix, rb, cb) for rb, cb in zip(rblocks, cblocks)
    ]
    rhs_submatrices = [
        extract_submatrix(rhs, rb, rhs_coords) for rb in rblocks
    ]

    n_blocks = len(rblocks)

    sol_submatrices = []
    for i in range(n_blocks):
        lhs = lhs_submatrices[i]
        incident_blocks = [
            (j, extract_submatrix(matrix, rblocks[i], cblocks[j]))
            for j in dag[i] if j != i
        ]

        rhs_terms = [rhs_submatrices[i]]
        rhs_terms.extend(
            -structure_preserving_product(block, sol_submatrices[j])
            for (j, block) in incident_blocks
        )
        sum_helper = CondensedSparseSummation(rhs_terms)
        rhs = sum_helper.sum(rhs_terms)

        factor = sps.linalg.splu(lhs)
        rhs_csc = rhs.tocsc()
        rhs_mask = (rhs_csc.indptr[rhs_coords] < rhs_csc.indptr[rhs_coords + 1])
        # This array maps column coordinates of the compressed matrix
        # to column coordinates in the full matrix
        compressed_to_full = np.arange(nrhs)[rhs_mask]

        if np.any(rhs_mask):
            compressed_rhs = sps.hstack(
                tuple(rhs_csc.getcol(j) for j in range(nrhs) if rhs_mask[j])
            )

            sol_i_compressed = factor.solve(compressed_rhs.toarray())
            sol_i_compressed = _dense_to_full_sparse(sol_i_compressed)

            #
            # "Expand" by mapping column coordinates back to the "full space"
            #
            sol_i_row = sol_i_compressed.row
            sol_i_col = compressed_to_full[sol_i_compressed.col]
            sol_i_data = sol_i_compressed.data
            sol_i = sps.coo_matrix(
                (sol_i_data, (sol_i_row, sol_i_col)),
                shape=(len(rblocks[i]), nrhs),
            )
        else:
            sol_i = sps.coo_matrix((len(rblocks[i]), nrhs))

        sol_submatrices.append(sol_i)

    # This is the solution in the row-permuted space
    perm_sol = sps.vstack(sol_submatrices).tocoo()

    # We apply to the solution rows the column permutation we applied to the
    # original matrix.
    row_perm_to_orig = np.array(sum(cblocks, []))

    sol_row = row_perm_to_orig[perm_sol.row]
    sol_col = perm_sol.col
    sol_data = perm_sol.data
    sol = sps.coo_matrix((sol_data, (sol_row, sol_col)), shape=(nrow, nrhs))
    return sol


def get_coords_to_extract(matrix, rows, cols):
    matrix = matrix.tocoo()
    nrow, ncol = matrix.shape
    sub_nrow = len(rows)
    sub_ncol = len(cols)
    mask = np.isin(matrix.row, rows)
    mask &= np.isin(matrix.col, cols)

    row_old_to_new = -np.ones(nrow)
    # Replace coordinates we want to extract with their location in the
    # provided coordinate arrays.
    row_old_to_new[rows] = np.arange(sub_nrow)
    col_old_to_new = -np.ones(ncol)
    col_old_to_new[cols] = np.arange(sub_ncol)

    coords_to_extract = np.arange(matrix.nnz)[mask]

    # Extract only the entries we want to keep
    sub_row = matrix.row[coords_to_extract]
    sub_col = matrix.col[coords_to_extract]

    # Put in coordinates of submatrix
    sub_row = row_old_to_new[sub_row]
    sub_col = col_old_to_new[sub_col]

    return sub_row, sub_col, coords_to_extract


class StructurePreservingSolver:
    """Linear solver for sparse systems that preserves all entries that
    can possibly be nonzero

    """
    def __init__(self, matrix, timer=None):
        """Initialize data structures based on matrix's sparsity structure"""
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer
        self._timer.start("symbolic")

        nrow, ncol = matrix.shape
        assert nrow == ncol
        self._dim = nrow

        rblocks, cblocks, dag = block_triangularize_with_dag(matrix)
        assert len(rblocks) == len(cblocks)
        self._n_blocks = len(rblocks)
        self._rblocks = rblocks
        self._cblocks = cblocks
        self._dag = dag

        #
        # Assemble row/col values for incident submatrices and coords to extract
        #
        self._incident_coords = [
            [
                (j, get_coords_to_extract(matrix, rblocks[i], cblocks[j]))
                for j in self._dag[i] if j != i
            ]
            for i in range(self._n_blocks)
        ]
        self._lhs_coords = [
            get_coords_to_extract(matrix, rblocks[i], cblocks[i])
            for i in range(self._n_blocks)
        ]

        # NOTE: This is the permutation of *solution* rows
        self._row_perm_to_orig = np.array(sum(self._cblocks, []))

        self._timer.stop("symbolic")

    # factorize_numeric depends on:
    # - self._rblocks, self._cblocks
    # - self._dag
    # - _incident_row, _incident_col, _incident_nz
    def factorize_numeric(self, matrix):
        """Factorize a matrix with the same sparsity structure as the original
        matrix

        """
        self._timer.start("numeric")
        #self._incident_blocks = [
        #    [
        #        (j, extract_submatrix(matrix, self._rblocks[i], self._cblocks[j]))
        #        for j in self._dag[i] if j != i
        #    ] 
        #    for i in range(self._n_blocks)
        #]
        self._incident_blocks = [
            [
                (
                    j,
                    sps.coo_matrix(
                        (matrix.data[coords_to_extract], (sub_row, sub_col)),
                        shape=(len(self._rblocks[i]), len(self._cblocks[j])),
                    ),
                )
                # Need to store j here so I can easily lookup coords later
                for j, (sub_row, sub_col, coords_to_extract) in incident
            ]
            for i, incident in enumerate(self._incident_coords)
        ]
        #lhs_submatrices = [
        #    # NOTE: This can be done faster given that sparsity structure
        #    # is cached
        #    extract_submatrix(matrix, rb, cb)
        #    for rb, cb in zip(self._rblocks, self._cblocks)
        #]
        lhs_submatrices = [
            sps.coo_matrix(
                (matrix.data[coords_to_extract], (sub_row, sub_col)),
                shape=(len(self._rblocks[i]), len(self._cblocks[i])),
            )
            for i, (sub_row, sub_col, coords_to_extract) in enumerate(self._lhs_coords)
        ]

        # TODO: Any way for splu to reuse symbolic factors (ordering)
        self._lhs_factors = [sps.linalg.splu(lhs.tocsc()) for lhs in lhs_submatrices]
        self._timer.stop("numeric")

    # solve depends on:
    # - self._rblocks, self._cblocks
    # - self._incident_blocks (as written, a combination of structural and numeric)
    # - self._lhs_factors (numeric)
    def solve(self, rhs):
        """Solve the linear system defined by the previously provided matrix
        and a right hand side

        """
        self._timer.start("backsolve")
        rhs_dim, nrhs = rhs.shape
        assert self._dim == rhs_dim
        rhs_coords = np.arange(nrhs)
        rhs_submatrices = [
            extract_submatrix(rhs, rb, rhs_coords) for rb in self._rblocks
        ]

        sol_submatrices = []
        for i in range(self._n_blocks):
            incident_blocks = self._incident_blocks[i]

            rhs_terms = [rhs_submatrices[i]]
            # TODO: Any performance benefit to re-using sparsity structure
            # of sums/products here? Is this even possible? If many RHS will
            # have same structure, then yes it is.
            rhs_terms.extend(
                -structure_preserving_product(block, sol_submatrices[j])
                for (j, block) in incident_blocks
            )
            sum_helper = CondensedSparseSummation(rhs_terms)
            # This is the RHS of the "subsystem" for this block
            rhs_i = sum_helper.sum(rhs_terms)

            factor = self._lhs_factors[i]

            rhs_csc = rhs_i.tocsc()
            # Compress columns of RHS (subblock) that are zero
            # NOTE: If we are repeatedly solving with RHSs with the same sparsity
            # pattern, this mask (and map) can be cached
            rhs_mask = (rhs_csc.indptr[rhs_coords] < rhs_csc.indptr[rhs_coords + 1])
            # This array maps column coordinates of the compressed matrix
            # to column coordinates in the full matrix
            compressed_to_full = rhs_coords[rhs_mask]

            if np.any(rhs_mask):
                compressed_rhs = sps.hstack(
                    tuple(rhs_csc.getcol(j) for j in range(nrhs) if rhs_mask[j])
                )

                # Solve to get a dense matrix
                sol_i_compressed = factor.solve(compressed_rhs.toarray())
                sol_i_compressed = _dense_to_full_sparse(sol_i_compressed)

                #
                # "Expand" by mapping column coordinates back to the "full space"
                #
                sol_i_row = sol_i_compressed.row
                sol_i_col = compressed_to_full[sol_i_compressed.col]
                sol_i_data = sol_i_compressed.data
                sol_i = sps.coo_matrix(
                    (sol_i_data, (sol_i_row, sol_i_col)),
                    shape=(len(self._rblocks[i]), nrhs),
                )
            else:
                # Solution to this block is an empty COO matrix
                sol_i = sps.coo_matrix((len(self._rblocks[i]), nrhs))

            sol_submatrices.append(sol_i)

        # This is the solution in the row-permuted space
        perm_sol = sps.vstack(sol_submatrices).tocoo()

        # We apply to the solution rows the column permutation we applied to the
        # original matrix.
        #
        # This flattening could be done previous to this method
        row_perm_to_orig = np.array(sum(self._cblocks, []))

        nrow = self._dim
        sol_row = row_perm_to_orig[perm_sol.row]
        sol_col = perm_sol.col
        sol_data = perm_sol.data
        sol = sps.coo_matrix((sol_data, (sol_row, sol_col)), shape=(nrow, nrhs))
        self._timer.stop("backsolve")
        return sol

    def symbolic_backsolve(self, rhs):
        self._timer.start("symbolic-backsolve")
        rhs_dim, nrhs = rhs.shape
        assert self._dim == rhs_dim
        rhs_coords = np.arange(nrhs)
        self._rhs_sub_coords = [
            get_coords_to_extract(rhs, rb, rhs_coords) for rb in self._rblocks
        ]

        # Get submatrices to use to build masks.
        # TODO: More efficient way to do this?
        rhs_submatrices = [
            sps.coo_matrix(
                (rhs.data[to_extract], (sub_row, sub_col)),
                shape=(len(self._rblocks[i]), nrhs),
            )
            for i, (sub_row, sub_col, to_extract) in enumerate(self._rhs_sub_coords)
        ]
        rhs_csc = [sm.tocsc() for sm in rhs_submatrices]
        self._rhs_masks = [
            (sm.indptr[rhs_coords] < sm.indptr[rhs_coords + 1])
            for sm in rhs_csc
        ]
        # This array maps column coordinates of the compressed matrix
        # to column coordinates in the full matrix
        self._compressed_to_full = [rhs_coords[mask] for mask in self._rhs_masks]
        self._timer.stop("symbolic-backsolve")

    def numeric_backsolve(self, rhs):
        self._timer.start("numeric-backsolve")
        rhs_dim, nrhs = rhs.shape
        assert self._dim == rhs_dim
        rhs_coords = np.arange(nrhs)
        rhs_submatrices = [
            sps.coo_matrix(
                (rhs.data[to_extract], (sub_row, sub_col)),
                shape=(len(self._rblocks[i]), nrhs),
            )
            for i, (sub_row, sub_col, to_extract) in enumerate(self._rhs_sub_coords)
        ]

        # TODO: Any alternative to building up this list sequentially?
        sol_submatrices = []
        for i in range(self._n_blocks):
            rhs_terms = [rhs_submatrices[i]]
            # TODO: Cache sparsity structure for sum/product
            rhs_terms.extend(
                -structure_preserving_product(block, sol_submatrices[j])
                for (j, block) in self._incident_blocks[i]
            )
            sum_helper = CondensedSparseSummation(rhs_terms)
            # This is the RHS of the "subsystem" for this block
            rhs_i = sum_helper.sum(rhs_terms)

            factor = self._lhs_factors[i]

            if np.any(self._rhs_masks[i]):
                rhs_csc = rhs_i.tocsc()
                # TODO: Most of this work should be doable in symbolic-backsolve
                compressed_rhs = sps.hstack(
                    tuple(
                        rhs_csc.getcol(j)
                        for j in range(nrhs)
                        if self._rhs_masks[i][j]
                    )
                )

                # Solve to get a dense matrix
                sol_i_compressed = factor.solve(compressed_rhs.toarray())
                # TODO: Should be possible to speed this up
                sol_i_compressed = _dense_to_full_sparse(sol_i_compressed)

                #
                # "Expand" by mapping column coordinates back to the "full space"
                #
                sol_i_row = sol_i_compressed.row
                # TODO: This should be doable in symbolic-backsolve
                sol_i_col = self._compressed_to_full[i][sol_i_compressed.col]
                sol_i_data = sol_i_compressed.data
                sol_i = sps.coo_matrix(
                    (sol_i_data, (sol_i_row, sol_i_col)),
                    shape=(len(self._rblocks[i]), nrhs),
                )
            else:
                sol_i = sps.coo_matrix((len(self._rblocks[i]), nrhs))

            sol_submatrices.append(sol_i)

        # TODO: Most of this work should be doable in structural-backsolve
        perm_sol = sps.vstack(sol_submatrices).tocoo()

        nrow = self._dim
        sol_row = self._row_perm_to_orig[perm_sol.row]
        sol_col = perm_sol.col
        sol_data = perm_sol.data
        sol = sps.coo_matrix((sol_data, (sol_row, sol_col)), shape=(nrow, nrhs))
        self._timer.stop("numeric-backsolve")
        return sol


def _structure_preserving_solve(matrix, rhs, timer=None):
    solver = StructurePreservingSolver(matrix, timer=timer)
    solver.factorize_numeric(matrix)
    sol = solver.solve(rhs)
    #solver.symbolic_backsolve(rhs)
    #sol = solver.numeric_backsolve(rhs)
    return sol
