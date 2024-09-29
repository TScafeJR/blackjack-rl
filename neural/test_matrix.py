import unittest

from .matrix import (add_row_vector, elementwise_add, elementwise_multiply,
                     matmul, matrix_shape, scale, transpose, zeros, zeros_like)


class TestMatrix(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(zeros(2, 3), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def test_zeros_like(self):
        self.assertEqual(zeros_like([[1.0, 2.0], [3.0, 4.0]]), [[0.0, 0.0], [0.0, 0.0]])

    def test_matrix_shape(self):
        self.assertEqual(matrix_shape([[1.0, 2.0, 3.0]]), (1, 3))

    def test_transpose(self):
        self.assertEqual(transpose([[1.0, 2.0], [3.0, 4.0]]), [[1.0, 3.0], [2.0, 4.0]])

    def test_matmul(self):
        result = matmul([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
        self.assertEqual(result, [[19.0, 22.0], [43.0, 50.0]])

    def test_matmul_dimension_mismatch(self):
        with self.assertRaises(Exception):
            matmul([[1.0, 2.0]], [[1.0, 2.0]])

    def test_add_row_vector(self):
        result = add_row_vector([[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0])
        self.assertEqual(result, [[11.0, 22.0], [13.0, 24.0]])

    def test_elementwise_add(self):
        result = elementwise_add([[1.0, 2.0]], [[3.0, 4.0]])
        self.assertEqual(result, [[4.0, 6.0]])

    def test_elementwise_multiply(self):
        result = elementwise_multiply([[2.0, 3.0]], [[4.0, 5.0]])
        self.assertEqual(result, [[8.0, 15.0]])

    def test_scale(self):
        self.assertEqual(scale([[1.0, -2.0]], 3.0), [[3.0, -6.0]])


if __name__ == "__main__":
    unittest.main()
