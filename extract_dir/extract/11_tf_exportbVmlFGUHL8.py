"/home/cc/Workspace/tfconstraint/python/ops/array_ops.py"
@tf_export(
    "linalg.matrix_transpose",
    v1=["linalg.transpose", "linalg.matrix_transpose", "matrix_transpose"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("matrix_transpose", "linalg.transpose")
def matrix_transpose(a, name="matrix_transpose", conjugate=False):
  """Transposes last two dimensions of tensor `a`.
  For example:
  ```python
  x = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.linalg.matrix_transpose(x)  # [[1, 4],
                                 #  [2, 5],
                                 #  [3, 6]]
  x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                   [4 + 4j, 5 + 5j, 6 + 6j]])
  tf.linalg.matrix_transpose(x, conjugate=True)  # [[1 - 1j, 4 - 4j],
                                                 #  [2 - 2j, 5 - 5j],
                                                 #  [3 - 3j, 6 - 6j]]
  # Matrix with two batch dimensions.
  # x.shape is [1, 2, 3, 4]
  # tf.linalg.matrix_transpose(x) is shape [1, 2, 4, 3]
  ```
  Note that `tf.matmul` provides kwargs allowing for transpose of arguments.
  This is done with minimal cost, and is preferable to using this function. E.g.
  ```python
  # Good!  Transpose is taken at minimal additional cost.
  tf.matmul(matrix, b, transpose_b=True)
  # Inefficient!
  tf.matmul(matrix, tf.linalg.matrix_transpose(b))
  ```
  @compatibility(numpy)
  In `numpy` transposes are memory-efficient constant time operations as they
  simply return a new view of the same data with adjusted `strides`.
  TensorFlow does not support strides, `linalg.matrix_transpose` returns a new
  tensor with the items permuted.
  @end_compatibility
  Args:
    a: A `Tensor` with `rank >= 2`.
    name: A name for the operation (optional).
    conjugate: Optional bool. Setting it to `True` is mathematically equivalent
      to tf.math.conj(tf.linalg.matrix_transpose(input)).
  Returns:
    A transposed batch matrix `Tensor`.
  Raises:
    ValueError:  If `a` is determined statically to have `rank < 2`.
  """
  with ops.name_scope(name, values=[a]):
    a = ops.convert_to_tensor(a, name="a")
    # If we know the number of dimensions (statically), we can do two things:
    # 1. Check that `a` is a (batch) matrix.
    # 2. Use a Python list for perm.  This preserves static shape information
    #    and avoids extra computations.
    a_shape = a.get_shape()
    ndims = a_shape.ndims
    if ndims is not None:
      if ndims < 2:
        raise ValueError("Argument `a` should be a (batch) matrix with rank "
                         f">= 2.  Received `a` = {a} with shape: {a_shape}")
      perm = list(range(ndims - 2)) + [ndims - 1] + [ndims - 2]
    else:
      a_rank = rank(a)
      perm = concat(
          (gen_math_ops._range(0, a_rank - 2, 1), [a_rank - 1, a_rank - 2]), 0)
    return transpose(a, perm=perm, conjugate=conjugate)
