@tf_export("random.stateless_uniform")
@dispatch.add_dispatch_support
def stateless_random_uniform(shape,
                             seed,
                             minval=0,
                             maxval=None,
                             dtype=dtypes.float32,
                             name=None,
                             alg="auto_select"):
  """Outputs deterministic pseudorandom values from a uniform distribution.
  This is a stateless version of `tf.random.uniform`: if run twice with the
  same seeds and shapes, it will produce the same pseudorandom numbers.  The
  output is consistent across multiple runs on the same hardware (and between
  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.
  The generated values follow a uniform distribution in the range
  `[minval, maxval)`. The lower bound `minval` is included in the range, while
  the upper bound `maxval` is excluded.
  For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
  be specified explicitly.
  In the integer case, the random integers are slightly biased unless
  `maxval - minval` is an exact power of two.  The bias is small for values of
  `maxval - minval` significantly smaller than the range of the output (either
  `2**32` or `2**64`).
  For full-range (i.e. inclusive of both max and min) random integers, pass
  `minval=None` and `maxval=None` with an integer `dtype`. For an integer dtype
  either both `minval` and `maxval` must be `None` or neither may be `None`. For
  example:
  ```python
  ints = tf.random.stateless_uniform(
      [10], seed=(2, 3), minval=None, maxval=None, dtype=tf.int32)
  ```
  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    minval: A Tensor or Python value of type `dtype`, broadcastable with
      `shape` (for integer types, broadcasting is not supported, so it needs to
      be a scalar). The lower bound on the range of random values to
      generate. Pass `None` for full-range integers.  Defaults to 0.
    maxval: A Tensor or Python value of type `dtype`, broadcastable with
      `shape` (for integer types, broadcasting is not supported, so it needs to
      be a scalar). The upper bound on the range of random values to generate.
      Defaults to 1 if `dtype` is floating point. Pass `None` for full-range
      integers.
    dtype: The type of the output: `float16`, `bfloat16`, `float32`, `float64`,
      `int32`, or `int64`. For unbounded uniform ints (`minval`, `maxval` both
      `None`), `uint32` and `uint64` may be used. Defaults to `float32`.
    name: A name for the operation (optional).
    alg: The RNG algorithm used to generate the random numbers. Valid
      choices are `"philox"` for [the Philox
      algorithm](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf),
      `"threefry"` for [the ThreeFry
      algorithm](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf),
      and `"auto_select"` (default) for the system to automatically
      select an algorithm based the device type. Values of
      `tf.random.Algorithm` can also be used. Note that with
      `"auto_select"`, the outputs of this function may change when
      it is running on a different device.
  Returns:
    A tensor of the specified shape filled with random uniform values.
  Raises:
    ValueError: If `dtype` is integral and only one of `minval` or `maxval` is
      specified.
  """
  dtype = dtypes.as_dtype(dtype)
  accepted_dtypes = (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                     dtypes.float64, dtypes.int32, dtypes.int64, dtypes.uint32,
                     dtypes.uint64)
  if dtype not in accepted_dtypes:
    raise ValueError(
        f"Argument `dtype` got invalid value {dtype}. Accepted dtypes are "
        f"{accepted_dtypes}.")
  if dtype.is_integer:
    if (minval is None) != (maxval is None):
      raise ValueError(
          f"For integer `dtype` argument {dtype}, argument `minval` and "
          f"`maxval` must be both None or not None. Got `minval`={minval} and "
          f"`maxval`={maxval}.")
    if minval is not None and dtype in (dtypes.uint32, dtypes.uint64):
      raise ValueError(
          f"Argument `dtype` got invalid value {dtype} when argument `minval` "
          f"is not None. Please don't use unsigned integers in this case.")
  elif maxval is None:
    maxval = 1
  with ops.name_scope(name, "stateless_random_uniform",
                      [shape, seed, minval, maxval]) as name:
    shape = shape_util.shape_tensor(shape)
    if dtype.is_integer and minval is None:
      key, counter, alg = _get_key_counter_alg(seed, alg)
      result = (
          gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2(
              shape, key=key, counter=counter, dtype=dtype, alg=alg, name=name))
    else:
      minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
      maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
      if dtype.is_integer:
        key, counter, alg = _get_key_counter_alg(seed, alg)
        result = gen_stateless_random_ops_v2.stateless_random_uniform_int_v2(
            shape,
            key=key,
            counter=counter,
            minval=minval,
            maxval=maxval,
            alg=alg,
            name=name)
      else:
        key, counter, alg = _get_key_counter_alg(seed, alg)
        rnd = gen_stateless_random_ops_v2.stateless_random_uniform_v2(
            shape, key=key, counter=counter, dtype=dtype, alg=alg)
        result = math_ops.add(rnd * (maxval - minval), minval, name=name)
    shape_util.maybe_set_static_shape(result, shape)
    return result
