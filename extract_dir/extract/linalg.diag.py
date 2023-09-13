@tf_export("linalg.diag", v1=["linalg.diag", "matrix_diag"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("matrix_diag")
def matrix_diag(diagonal,
                name="diag",
                k=0,
                num_rows=-1,
                num_cols=-1,
                padding_value=0,
                align="RIGHT_LEFT"):
  """Returns a batched diagonal tensor with given batched diagonal values.
  Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
  diagonals of a matrix, with everything else padded with `padding`. `num_rows`
  and `num_cols` specify the dimension of the innermost matrix of the output. If
  both are not specified, the op assumes the innermost matrix is square and
  infers its size from `k` and the innermost dimension of `diagonal`. If only
  one of them is specified, the op assumes the unspecified value is the smallest
  possible based on other criteria.
  Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor
  has rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only
  one diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has
  rank `r` with shape `[I, J, ..., L, num_rows, num_cols]`.
  The second innermost dimension of `diagonal` has double meaning. When `k` is
  scalar or `k[0] == k[1]`, `M` is part of the batch size [I, J, ..., M], and
  the output tensor is:
  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
      padding_value                             ; otherwise
  ```
  Otherwise, `M` is treated as the number of diagonals for the matrix in the
  same batch (`M = k[1]-k[0]+1`), and the output tensor is:
  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
      padding_value                                     ; otherwise
  ```
  where `d = n - m`, `diag_index = k[1] - d`, and
  `index_in_diag = n - max(d, 0) + offset`.
  `offset` is zero except when the alignment of the diagonal is to the right.
  ```
  offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                             and `d >= 0`) or
                                           (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                             and `d <= 0`)
           0                          ; otherwise
  ```
  where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.
  For example:
  ```
  # The main diagonal.
  diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
                       [5, 6, 7, 8]])
  tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
                                 [0, 2, 0, 0],
                                 [0, 0, 3, 0],
                                 [0, 0, 0, 4]],
                                [[5, 0, 0, 0],
                                 [0, 6, 0, 0],
                                 [0, 0, 7, 0],
                                 [0, 0, 0, 8]]]
  # A superdiagonal (per batch).
  diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
                       [4, 5, 6]])
  tf.matrix_diag(diagonal, k = 1)
    ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
          [0, 0, 2, 0],
          [0, 0, 0, 3],
          [0, 0, 0, 0]],
         [[0, 4, 0, 0],
          [0, 0, 5, 0],
          [0, 0, 0, 6],
          [0, 0, 0, 0]]]
  # A tridiagonal band (per batch).
  diagonals = np.array([[[8, 9, 0],  # Input shape: (2, 2, 3)
                         [1, 2, 3],
                         [0, 4, 5]],
                        [[2, 3, 0],
                         [6, 7, 9],
                         [0, 9, 1]]])
  tf.matrix_diag(diagonals, k = (-1, 1))
    ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
          [4, 2, 9],
          [0, 5, 3]],
         [[6, 2, 0],
          [9, 7, 3],
          [0, 1, 9]]]
  # RIGHT_LEFT alignment.
  diagonals = np.array([[[0, 8, 9],  # Input shape: (2, 2, 3)
                         [1, 2, 3],
                         [4, 5, 0]],
                        [[0, 2, 3],
                         [6, 7, 9],
                         [9, 1, 0]]])
  tf.matrix_diag(diagonals, k = (-1, 1), align="RIGHT_LEFT")
    ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
          [4, 2, 9],
          [0, 5, 3]],
         [[6, 2, 0],
          [9, 7, 3],
          [0, 1, 9]]]
  # Rectangular matrix.
  diagonal = np.array([1, 2])  # Input shape: (2)
  tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
    ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
         [1, 0, 0, 0],
         [0, 2, 0, 0]]
  # Rectangular matrix with inferred num_cols and padding_value = 9.
  tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
    ==> [[9, 9],  # Output shape: (3, 2)
         [1, 9],
         [9, 2]]
  ```
  Args:
    diagonal: A `Tensor` with `rank k >= 1`.
    name: A name for the operation (optional).
    k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the
      main diagonal, and negative value means subdiagonals. `k` can be a single
      integer (for a single diagonal) or a pair of integers specifying the low
      and high ends of a matrix band. `k[0]` must not be larger than `k[1]`.
    num_rows: The number of rows of the output matrix. If it is not provided,
      the op assumes the output matrix is a square matrix and infers the matrix
      size from `d_lower`, `d_upper`, and the innermost dimension of `diagonal`.
    num_cols: The number of columns of the output matrix. If it is not provided,
      the op assumes the output matrix is a square matrix and infers the matrix
      size from `d_lower`, `d_upper`, and the innermost dimension of `diagonal`.
    padding_value: The value to fill the area outside the specified diagonal
      band with. Default is 0.
    align: Some diagonals are shorter than `max_diag_len` and need to be padded.
      `align` is a string specifying how superdiagonals and subdiagonals should
      be aligned, respectively. There are four possible alignments: "RIGHT_LEFT"
      (default), "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT"
      aligns superdiagonals to the right (left-pads the row) and subdiagonals to
      the left (right-pads the row). It is the packing format LAPACK uses.
      cuSPARSE uses "LEFT_RIGHT", which is the opposite alignment.
  Returns:
    A Tensor. Has the same type as `diagonal`.
  """
  # Special case to sidestep the tf.constant conversion error:
  # TypeError: Expected bool, got 0 of type 'int' instead.
  if hasattr(diagonal, "dtype") and diagonal.dtype == "bool":
    padding_value = bool(padding_value)
  return gen_array_ops.matrix_diag_v3(
      diagonal=diagonal,
      k=k,
      num_rows=num_rows,
      num_cols=num_cols,
      padding_value=padding_value,
      align=align,
      name=name)
