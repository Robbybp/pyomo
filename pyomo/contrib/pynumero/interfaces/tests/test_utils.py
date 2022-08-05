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

import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

import pyomo.contrib.pynumero.interfaces.utils as utils


class TestCondensedSparseSummation(unittest.TestCase):
    def test_condensed_sparse_summation(self):
        data = [1.0, 0.0]
        row = [1, 2]
        col = [2, 2]
        A = scipy.sparse.coo_matrix( (data, (row,col)), shape=(3,3) )

        data = [3.0, 0.0]
        B = scipy.sparse.coo_matrix( (data, (row,col)), shape=(3,3) )

        # By default, scipy will remove structural nonzeros that
        # have zero values
        C = A + B
        self.assertEqual(C.nnz, 1)

        # Our CondensedSparseSummation should not remove any
        # structural nonzeros
        sparse_sum = utils.CondensedSparseSummation([A,B])
        C = sparse_sum.sum([A,B])
        expected_data = np.asarray([4.0, 0.0], dtype=np.float64)
        expected_row = np.asarray([1, 2], dtype=np.int64)
        expected_col = np.asarray([2, 2], dtype=np.int64)
        self.assertTrue(np.array_equal(expected_data, C.data))
        self.assertTrue(np.array_equal(expected_row, C.row))
        self.assertTrue(np.array_equal(expected_col, C.col))

        B.data[1] = 5.0
        C = sparse_sum.sum([A,B])
        expected_data = np.asarray([4.0, 5.0], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_data, C.data))
        self.assertTrue(np.array_equal(expected_row, C.row))
        self.assertTrue(np.array_equal(expected_col, C.col))

        B.data[1] = 0.0
        C = sparse_sum.sum([A,B])
        expected_data = np.asarray([4.0, 0.0], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_data, C.data))
        self.assertTrue(np.array_equal(expected_row, C.row))
        self.assertTrue(np.array_equal(expected_col, C.col))

    def test_repeated_row_col(self):
        data = [1.0, 0.0, 2.0]
        row = [1, 2, 1]
        col = [2, 2, 2]
        A = scipy.sparse.coo_matrix( (data, (row,col)), shape=(3,3) )

        data = [3.0, 0.0]
        row = [1, 2]
        col = [2, 2]
        B = scipy.sparse.coo_matrix( (data, (row,col)), shape=(3,3) )

        # Our CondensedSparseSummation should not remove any
        # structural nonzeros
        sparse_sum = utils.CondensedSparseSummation([A,B])
        C = sparse_sum.sum([A,B])
        expected_data = np.asarray([6.0, 0.0], dtype=np.float64)
        expected_row = np.asarray([1, 2], dtype=np.int64)
        expected_col = np.asarray([2, 2], dtype=np.int64)
        self.assertTrue(np.array_equal(expected_data, C.data))
        self.assertTrue(np.array_equal(expected_row, C.row))
        self.assertTrue(np.array_equal(expected_col, C.col))


class TestSparseMultiply(unittest.TestCase):

    def test_multiply_sparsity_structure(self):
        sps = scipy.sparse

        row1 = np.array([1, 1])
        col1 = np.array([0, 1])
        data1 = np.array([0, 0])
        matrix1 = sps.coo_matrix((data1, (row1, col1)), shape=(2, 2))

        row2 = np.array([0, 0, 1])
        col2 = np.array([0, 1, 2])
        data2 = np.array([0, 0, 0])
        matrix2 = sps.coo_matrix((data2, (row2, col2)), shape=(2, 3))

        row, col = utils.multiply_sparsity_structure(matrix1, matrix2)
        rc_set = set(zip(row, col))

        pred_row = np.array([1, 1, 1])
        pred_col = np.array([0, 1, 2])    
        pred_rc_set = set(zip(pred_row, pred_col))

        # We assert nothing about the order of the nonzeros
        self.assertEqual(rc_set, pred_rc_set)

    def test_augment_sparsity_structure(self):
        sps = scipy.sparse

        row1 = np.array([1, 1])
        col1 = np.array([0, 1])
        data1 = np.array([0, 0])
        matrix1 = sps.coo_matrix((data1, (row1, col1)), shape=(2, 2))

        row2 = np.array([0, 0, 1])
        col2 = np.array([0, 1, 2])
        data2 = np.array([1, 2, 3])
        matrix2 = sps.coo_matrix((data2, (row2, col2)), shape=(2, 3))

        product = matrix1.dot(matrix2).tocoo()
        row = np.array([1, 1, 1])
        col = np.array([0, 1, 2])

        # If we update SciPy and this test starts failing, maybe
        # they have "fixed" this feature.
        self.assertEqual(product.nnz, 0)

        product = utils.augment_sparsity_structure(product, row, col)
        # Order is maintained among added entries.
        np.testing.assert_array_equal(product.row, row)
        np.testing.assert_array_equal(product.col, col)

    def test_augment_sparsity_structure_with_cancellation(self):
        sps = scipy.sparse

        row1 = np.array([0, 0, 1])
        col1 = np.array([0, 1, 1])
        data1 = np.array([1, -1, 2])
        matrix1 = sps.coo_matrix((data1, (row1, col1)), shape=(2, 2))

        row2 = np.array([0, 0, 1])
        col2 = np.array([0, 1, 1])
        data2 = np.array([3, 2, 2])
        matrix2 = sps.coo_matrix((data2, (row2, col2)), shape=(2, 2))

        product = matrix1.dot(matrix2).tocoo()
        # If we update SciPy and this test starts failing, maybe
        # they have "fixed" this feature.
        self.assertEqual(product.nnz, 2)

        product = utils.augment_sparsity_structure(product, row1, col1)
        pred_row = np.array([0, 1, 0])
        pred_col = np.array([0, 1, 1])
        np.testing.assert_array_equal(product.row, pred_row)
        np.testing.assert_array_equal(product.col, pred_col)

    def test_sparse_multiply(self):
        sps = scipy.sparse

        matrix1 = sps.identity(5).tocoo()

        row2 = np.array([0, 1, 2, 3, 4])
        col2 = np.array([0, 1, 2, 3, 4])
        data2 = np.array([1, 0, 3, 0, 5])
        matrix2 = sps.coo_matrix((data2, (row2, col2)), shape=(5, 5))

        product = utils.safe_coo_multiply(matrix1, matrix2)

        pred_nz_set = set(zip(row2, col2))
        nz_set = set(zip(product.row, product.col))
        self.assertEqual(pred_nz_set, nz_set)
        np.testing.assert_array_equal(matrix2.toarray(), product.toarray())

    def test_sparse_multiply_with_cancellation(self):
        sps = scipy.sparse

        row1 = np.array([0, 0, 1])
        col1 = np.array([0, 1, 1])
        data1 = np.array([1, -1, 2])
        matrix1 = sps.coo_matrix((data1, (row1, col1)), shape=(2, 2))

        row2 = np.array([0, 0, 1])
        col2 = np.array([0, 1, 1])
        data2 = np.array([3, 2, 2])
        matrix2 = sps.coo_matrix((data2, (row2, col2)), shape=(2, 2))

        product = utils.safe_coo_multiply(matrix1, matrix2)
        self.assertEqual(product.nnz, 3)

        row = np.array([0, 1, 0])
        col = np.array([0, 1, 1])
        data = np.array([3, 4, 0])
        pred_prod = sps.coo_matrix((data, (row, col)), shape=(2, 2))
        nz_set = set(zip(product.row, product.col))
        pred_nz_set = set(zip(row, col))
        self.assertEqual(pred_nz_set, nz_set)
        np.testing.assert_array_equal(product.toarray(), pred_prod.toarray())


if __name__ == '__main__':
    TestCondensedSparseSummation().test_condensed_sparse_summation()
