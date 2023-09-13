@tf_export("random.uniform", v1=["random.uniform", "random_uniform"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_uniform")
def random_uniform(shape,
                   minval=0,
                   maxval=None,
                   dtype=dtypes.float32,
                   seed=None,
                   name=None):
  """Outputs random values from a uniform distribution.
  The generated values follow a uniform distribution in the range
  `[minval, maxval)`. The lower bound `minval` is included in the range, while
  the upper bound `maxval` is excluded.
  For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
  be specified explicitly.
  In the integer case, the random integers are slightly biased unless
  `maxval - minval` is an exact power of two.  The bias is small for values of
  `maxval - minval` significantly smaller than the range of the output (either
  `2**32` or `2**64`).
  Examples:
  >>> tf.random.uniform(shape=[2])
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([..., ...], dtype=float32)>
  >>> tf.random.uniform(shape=[], minval=-1., maxval=0.)
  <tf.Tensor: shape=(), dtype=float32, numpy=-...>
  >>> tf.random.uniform(shape=[], minval=5, maxval=10, dtype=tf.int64)
  <tf.Tensor: shape=(), dtype=int64, numpy=...>
  The `seed` argument produces a deterministic sequence of tensors across
  multiple calls. To repeat that sequence, use `tf.random.set_seed`:
  >>> tf.random.set_seed(5)
  >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
  <tf.Tensor: shape=(), dtype=int32, numpy=2>
  >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
  <tf.Tensor: shape=(), dtype=int32, numpy=0>
  >>> tf.random.set_seed(5)
  >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
  <tf.Tensor: shape=(), dtype=int32, numpy=2>
  >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
  <tf.Tensor: shape=(), dtype=int32, numpy=0>
  Without `tf.random.set_seed` but with a `seed` argument is specified, small
  changes to function graphs or previously executed operations will change the
  returned value. See `tf.random.set_seed` for details.
  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    minval: A Tensor or Python value of type `dtype`, broadcastable with
      `shape` (for integer types, broadcasting is not supported, so it needs to
      be a scalar). The lower bound on the range of random values to generate
      (inclusive).  Defaults to 0.
    maxval: A Tensor or Python value of type `dtype`, broadcastable with
      `shape` (for integer types, broadcasting is not supported, so it needs to
      be a scalar). The upper bound on the range of random values to generate
      (exclusive). Defaults to 1 if `dtype` is floating point.
    dtype: The type of the output: `float16`, `bfloat16`, `float32`, `float64`,
      `int32`, or `int64`. Defaults to `float32`.
    seed: A Python integer. Used in combination with `tf.random.set_seed` to
      create a reproducible sequence of tensors across multiple calls.
    name: A name for the operation (optional).
  Returns:
    A tensor of the specified shape filled with random uniform values.
  Raises:
    ValueError: If `dtype` is integral and `maxval` is not specified.
  """
  dtype = dtypes.as_dtype(dtype)
  accepted_dtypes = (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                     dtypes.float64, dtypes.int32, dtypes.int64)
  if dtype not in accepted_dtypes:
    raise ValueError(
        f"Argument `dtype` got invalid value {dtype}. Accepted dtypes are "
        f"{accepted_dtypes}.")
  if maxval is None:
    if dtype.is_integer:
      raise ValueError("Must specify maxval for integer dtype %r" % dtype)
    maxval = 1
  with ops.name_scope(name, "random_uniform", [shape, minval, maxval]) as name:
    shape = shape_util.shape_tensor(shape)
    # In case of [0,1) floating results, minval and maxval is unused. We do an
    # `is` comparison here since this is cheaper than isinstance or  __eq__.
    minval_is_zero = isinstance(minval, int) and minval == 0
    maxval_is_one = isinstance(maxval, int) and maxval == 1
    if not minval_is_zero or not maxval_is_one or dtype.is_integer:
      minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
      maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
    seed1, seed2 = random_seed.get_seed(seed)
    if dtype.is_integer:
      result = gen_random_ops.random_uniform_int(
          shape, minval, maxval, seed=seed1, seed2=seed2, name=name)
    else:
      result = gen_random_ops.random_uniform(
          shape, dtype, seed=seed1, seed2=seed2)
      if minval_is_zero:
        if not maxval_is_one:
          result = math_ops.multiply(result, maxval)
      else:
        result = math_ops.add(result * (maxval - minval), minval, name=name)
    # TODO(b/132092188): C++ shape inference inside functional ops does not
    # cross FuncGraph boundaries since that information is only available in
    # python. So we manually get the static shape using
    # `constant_value_as_shape` which *does* cross function boundaries.
    shape_util.maybe_set_static_shape(result, shape)
    return result
