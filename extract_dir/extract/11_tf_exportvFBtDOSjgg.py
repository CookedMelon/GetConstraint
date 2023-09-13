"/home/cc/Workspace/tfconstraint/python/ops/linalg_ops.py"
@tf_export(
    'linalg.triangular_solve',
    v1=['linalg.triangular_solve', 'matrix_triangular_solve'])
@dispatch.add_dispatch_support
def matrix_triangular_solve(matrix, rhs, lower=True, adjoint=False, name=None):
  """Solve systems of linear equations with upper or lower triangular matrices.
  `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
  square matrices. If `lower` is `True` then the strictly upper triangular part
  of each inner-most matrix is assumed to be zero and not accessed. If `lower`
  is `False` then the strictly lower triangular part of each inner-most matrix
  is assumed to be zero and not accessed. `rhs` is a tensor of shape
  `[..., M, N]`.
  The output is a tensor of shape `[..., M, N]`. If `adjoint` is `True` then the
  innermost matrices in output satisfy matrix equations `
  sum_k matrix[..., i, k] * output[..., k, j] = rhs[..., i, j]`.
  If `adjoint` is `False` then the
  innermost matrices in output satisfy matrix equations
  `sum_k adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.
  Example:
  >>> a = tf.constant([[3,  0,  0,  0],
  ...   [2,  1,  0,  0],
  ...   [1,  0,  1,  0],
  ...   [1,  1,  1,  1]], dtype=tf.float32)
  >>> b = tf.constant([[4], [2], [4], [2]], dtype=tf.float32)
  >>> x = tf.linalg.triangular_solve(a, b, lower=True)
  >>> x
  <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
  array([[ 1.3333334 ],
         [-0.66666675],
         [ 2.6666665 ],
         [-1.3333331 ]], dtype=float32)>
  >>> tf.matmul(a, x)
  <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
  array([[4.],
         [2.],
         [4.],
         [2.]], dtype=float32)>
  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`,
      `float32`, `half`, `complex64`, `complex128`. Shape is `[..., M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`. Shape is `[..., M,
      N]`.
    lower: An optional `bool`. Defaults to `True`. Boolean indicating whether
      the innermost matrices in matrix are lower or upper triangular.
    adjoint: An optional `bool`. Defaults to `False`. Boolean indicating whether
      to solve with matrix or its (block-wise) adjoint.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as matrix, and shape is `[..., M, N]`.
  """
  with ops.name_scope(name, 'triangular_solve', [matrix, rhs]):
    return gen_linalg_ops.matrix_triangular_solve(
        matrix, rhs, lower=lower, adjoint=adjoint)
