from typing import List, Tuple


def zeros(num_rows: int, num_cols: int) -> List[List[float]]:
    return [[0.0] * num_cols for _ in range(num_rows)]


def zeros_like(matrix: List[List[float]]) -> List[List[float]]:
    return zeros(len(matrix), len(matrix[0]))


def matrix_shape(matrix: List[List[float]]) -> Tuple[int, int]:
    return len(matrix), len(matrix[0])


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [list(column) for column in zip(*matrix)]


def matmul(
    matrix_a: List[List[float]], matrix_b: List[List[float]]
) -> List[List[float]]:
    if len(matrix_a[0]) != len(matrix_b):
        raise Exception("Matrix dimensions do not align")

    columns_b = transpose(matrix_b)
    return [
        [sum(a * b for a, b in zip(row, column)) for column in columns_b]
        for row in matrix_a
    ]


def add_row_vector(
    matrix: List[List[float]], row_vector: List[float]
) -> List[List[float]]:
    return [
        [value + offset for value, offset in zip(row, row_vector)] for row in matrix
    ]


def elementwise_add(
    matrix_a: List[List[float]], matrix_b: List[List[float]]
) -> List[List[float]]:
    return [
        [a + b for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(matrix_a, matrix_b)
    ]


def elementwise_multiply(
    matrix_a: List[List[float]], matrix_b: List[List[float]]
) -> List[List[float]]:
    return [
        [a * b for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(matrix_a, matrix_b)
    ]


def scale(matrix: List[List[float]], factor: float) -> List[List[float]]:
    return [[value * factor for value in row] for row in matrix]
