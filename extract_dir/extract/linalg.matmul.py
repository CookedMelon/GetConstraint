@tf_export("linalg.matmul", "matmul")
@dispatch.add_dispatch_support
def matmul(a,
           b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           a_is_sparse=False,
           b_is_sparse=False,
           output_type=None,
           name=None):
  """Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
  The inputs must, following any transpositions, be tensors of rank >= 2
  where the inner 2 dimensions specify valid matrix multiplication dimensions,
  and any further outer dimensions specify matching batch size.
  Both matrices must be of the same type. The supported types are:
  `bfloat16`, `float16`, `float32`, `float64`, `int32`, `int64`,
  `complex64`, `complex128`.
  Either matrix can be transposed or adjointed (conjugated and transposed) on
  the fly by setting one of the corresponding flag to `True`. These are `False`
  by default.
  If one or both of the matrices contain a lot of zeros, a more efficient
  multiplication algorithm can be used by setting the corresponding
  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
  This optimization is only available for plain matrices (rank-2 tensors) with
  datatypes `bfloat16` or `float32`.
  A simple 2-D tensor matrix multiplication:
  >>> a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
  >>> a  # 2-D tensor
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[1, 2, 3],
         [4, 5, 6]], dtype=int32)>
  >>> b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
  >>> b  # 2-D tensor
  <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
  array([[ 7,  8],
         [ 9, 10],
         [11, 12]], dtype=int32)>
  >>> c = tf.matmul(a, b)
  >>> c  # `a` * `b`
  <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
  array([[ 58,  64],
         [139, 154]], dtype=int32)>
  A batch matrix multiplication with batch shape [2]:
  >>> a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
  >>> a  # 3-D tensor
  <tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
  array([[[ 1,  2,  3],
          [ 4,  5,  6]],
         [[ 7,  8,  9],
          [10, 11, 12]]], dtype=int32)>
  >>> b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
  >>> b  # 3-D tensor
  <tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
  array([[[13, 14],
          [15, 16],
          [17, 18]],
         [[19, 20],
          [21, 22],
          [23, 24]]], dtype=int32)>
  >>> c = tf.matmul(a, b)
  >>> c  # `a` * `b`
  <tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
  array([[[ 94, 100],
          [229, 244]],
         [[508, 532],
          [697, 730]]], dtype=int32)>
  Since python >= 3.5 the @ operator is supported
  (see [PEP 465](https://www.python.org/dev/peps/pep-0465/)). In TensorFlow,
  it simply calls the `tf.matmul()` function, so the following lines are
  equivalent:
  >>> d = a @ b @ [[10], [11]]
  >>> d = tf.matmul(tf.matmul(a, b), [[10], [11]])
  Args:
    a: `tf.Tensor` of type `float16`, `float32`, `float64`, `int32`,
      `complex64`, `complex128` and rank > 1.
    b: `tf.Tensor` with same type and rank as `a`.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    adjoint_a: If `True`, `a` is conjugated and transposed before
      multiplication.
    adjoint_b: If `True`, `b` is conjugated and transposed before
      multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix. Notice, this
      **does not support `tf.sparse.SparseTensor`**, it just makes optimizations
      that assume most values in `a` are zero.
      See `tf.sparse.sparse_dense_matmul`
      for some support for `tf.sparse.SparseTensor` multiplication.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix. Notice, this
      **does not support `tf.sparse.SparseTensor`**, it just makes optimizations
      that assume most values in `b` are zero.
      See `tf.sparse.sparse_dense_matmul`
      for some support for `tf.sparse.SparseTensor` multiplication.
    output_type: The output datatype if needed. Defaults to None in which case
      the output_type is the same as input type. Currently only works when input
      tensors are type (u)int8 and output_type can be int32.
    name: Name for the operation (optional).
  Returns:
    A `tf.Tensor` of the same type as `a` and `b` where each inner-most matrix
    is the product of the corresponding matrices in `a` and `b`, e.g. if all
    transpose or adjoint attributes are `False`:
    `output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`,
    for all indices `i`, `j`.
    Note: This is matrix product, not element-wise product.
  Raises:
    ValueError: If `transpose_a` and `adjoint_a`, or `transpose_b` and
      `adjoint_b` are both set to `True`.
    TypeError: If output_type is specified but the types of `a`, `b` and
      `output_type` is not (u)int8, (u)int8 and int32.
  """
  with ops.name_scope(name, "MatMul", [a, b]) as name:
    if transpose_a and adjoint_a:
      raise ValueError(
          f"Only one of `transpose_a` and `adjoint_a` can be True. "
          f"Received `transpose_a`={transpose_a}, "
          f"`adjoint_a`={adjoint_a}.")
    if transpose_b and adjoint_b:
      raise ValueError(
          f"Only one of `transpose_b` and `adjoint_b` can be True. "
          f"Received `transpose_b`={transpose_b}, "
          f"`adjoint_b`={adjoint_b}.")
    if context.executing_eagerly():
      if not isinstance(a, (ops.EagerTensor, _resource_variable_type)):
        a = ops.convert_to_tensor(a, name="a")
      if not isinstance(b, (ops.EagerTensor, _resource_variable_type)):
        b = ops.convert_to_tensor(b, dtype_hint=a.dtype.base_dtype, name="b")
    else:
      a = ops.convert_to_tensor(a, name="a")
      b = ops.convert_to_tensor(b, dtype_hint=a.dtype.base_dtype, name="b")
    # TODO(apassos) remove _shape_tuple here when it is not needed.
    a_shape = a._shape_tuple()  # pylint: disable=protected-access
    b_shape = b._shape_tuple()  # pylint: disable=protected-access
    output_may_have_non_empty_batch_shape = (
        (a_shape is None or len(a_shape) > 2) or
        (b_shape is None or len(b_shape) > 2))
    # TODO(b/178749687): remove this boolean and all related branches once the
    # bridges are ready.
    # batch_matmul_v3 is for when input type is different from output type.
    use_batch_matmul_v3 = False
    if output_type and (output_type != a.dtype or output_type != b.dtype):
      use_batch_matmul_v3 = True
    if (not a_is_sparse and
        not b_is_sparse) and output_may_have_non_empty_batch_shape:
      # BatchMatmul does not support transpose, so we conjugate the matrix and
      # use adjoint instead. Conj() is a noop for real matrices.
      if transpose_a:
        a = conj(a)
        adjoint_a = True
      if transpose_b:
        b = conj(b)
        adjoint_b = True
      if use_batch_matmul_v3:
        return gen_math_ops.batch_mat_mul_v3(
            a, b, adj_x=adjoint_a, adj_y=adjoint_b, Tout=output_type, name=name)
      else:
        return gen_math_ops.batch_mat_mul_v2(
            a, b, adj_x=adjoint_a, adj_y=adjoint_b, name=name)
    # Neither matmul nor sparse_matmul support adjoint, so we conjugate
    # the matrix and use transpose instead. Conj() is a noop for real
    # matrices.
    if adjoint_a:
      a = conj(a)
      transpose_a = True
    if adjoint_b:
      b = conj(b)
      transpose_b = True
    use_sparse_matmul = False
    if a_is_sparse or b_is_sparse:
      sparse_matmul_types = [dtypes.bfloat16, dtypes.float32]
      use_sparse_matmul = (
          a.dtype in sparse_matmul_types and b.dtype in sparse_matmul_types)
    if (((a.dtype == dtypes.bfloat16 and
          b.dtype not in (dtypes.int8, dtypes.uint8)) or
         (b.dtype == dtypes.bfloat16 and
          a.dtype not in (dtypes.int8, dtypes.uint8))) and a.dtype != b.dtype):
      # matmul currently doesn't handle mixed-precision inputs other than
      # fp16 * int8 which is supported in BatchMatMulV3.
      use_sparse_matmul = True
    if use_sparse_matmul:
      ret = sparse_matmul(
          a,
          b,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          a_is_sparse=a_is_sparse,
          b_is_sparse=b_is_sparse,
          name=name)
      # sparse_matmul always returns float32, even with
      # bfloat16 inputs. This prevents us from configuring bfloat16 training.
      # casting to bfloat16 also matches non-sparse matmul behavior better.
      if a.dtype == dtypes.bfloat16 and b.dtype == dtypes.bfloat16:
        ret = cast(ret, dtypes.bfloat16)
      return ret
    else:
      if use_batch_matmul_v3:
        adjoint_a = adjoint_a or transpose_a
        adjoint_b = adjoint_b or transpose_b
        return gen_math_ops.batch_mat_mul_v3(
            a, b, adj_x=adjoint_a, adj_y=adjoint_b, Tout=output_type, name=name)
      else:
        return gen_math_ops.mat_mul(
            a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
