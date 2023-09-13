@tf_export("linalg.normalize")
@dispatch.add_dispatch_support
def normalize(tensor, ord="euclidean", axis=None, name=None):
  """Normalizes `tensor` along dimension `axis` using specified norm.
  This uses `tf.linalg.norm` to compute the norm along `axis`.
  This function can compute several different vector norms (the 1-norm, the
  Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0) and
  matrix norms (Frobenius, 1-norm, 2-norm and inf-norm).
  Args:
    tensor: `Tensor` of types `float32`, `float64`, `complex64`, `complex128`
    ord: Order of the norm. Supported values are `'fro'`, `'euclidean'`, `1`,
      `2`, `np.inf` and any positive real number yielding the corresponding
      p-norm. Default is `'euclidean'` which is equivalent to Frobenius norm if
      `tensor` is a matrix and equivalent to 2-norm for vectors.
      Some restrictions apply: a) The Frobenius norm `'fro'` is not defined for
        vectors, b) If axis is a 2-tuple (matrix norm), only `'euclidean'`,
        '`fro'`, `1`, `2`, `np.inf` are supported. See the description of `axis`
        on how to compute norms for a batch of vectors or matrices stored in a
        tensor.
    axis: If `axis` is `None` (the default), the input is considered a vector
      and a single vector norm is computed over the entire set of values in the
      tensor, i.e. `norm(tensor, ord=ord)` is equivalent to
      `norm(reshape(tensor, [-1]), ord=ord)`. If `axis` is a Python integer, the
      input is considered a batch of vectors, and `axis` determines the axis in
      `tensor` over which to compute vector norms. If `axis` is a 2-tuple of
      Python integers it is considered a batch of matrices and `axis` determines
      the axes in `tensor` over which to compute a matrix norm.
      Negative indices are supported. Example: If you are passing a tensor that
        can be either a matrix or a batch of matrices at runtime, pass
        `axis=[-2,-1]` instead of `axis=None` to make sure that matrix norms are
        computed.
    name: The name of the op.
  Returns:
    normalized: A normalized `Tensor` with the same shape as `tensor`.
    norm: The computed norms with the same shape and dtype `tensor` but the
      final axis is 1 instead. Same as running
      `tf.cast(tf.linalg.norm(tensor, ord, axis keepdims=True), tensor.dtype)`.
  Raises:
    ValueError: If `ord` or `axis` is invalid.
  """
  with ops.name_scope(name, "normalize", [tensor]) as name:
    tensor = ops.convert_to_tensor(tensor)
    norm = linalg_ops.norm(tensor, ord, axis, keepdims=True)
    norm = math_ops.cast(norm, tensor.dtype)
    normalized = tensor / norm
    return normalized, norm
