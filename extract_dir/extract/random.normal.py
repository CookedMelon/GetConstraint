@tf_export("random.normal", v1=["random.normal", "random_normal"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_normal")
def random_normal(shape,
                  mean=0.0,
                  stddev=1.0,
                  dtype=dtypes.float32,
                  seed=None,
                  name=None):
  """Outputs random values from a normal distribution.
  Example that generates a new set of random values every time:
  >>> tf.random.set_seed(5);
  >>> tf.random.normal([4], 0, 1, tf.float32)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=..., dtype=float32)>
  Example that outputs a reproducible result:
  >>> tf.random.set_seed(5);
  >>> tf.random.normal([2,2], 0, 1, tf.float32, seed=1)
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[-1.3768897 , -0.01258316],
        [-0.169515   ,  1.0824056 ]], dtype=float32)>
  In this case, we are setting both the global and operation-level seed to
  ensure this result is reproducible.  See `tf.random.set_seed` for more
  information.
  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean: A Tensor or Python value of type `dtype`, broadcastable with `stddev`.
      The mean of the normal distribution.
    stddev: A Tensor or Python value of type `dtype`, broadcastable with `mean`.
      The standard deviation of the normal distribution.
    dtype: The float type of the output: `float16`, `bfloat16`, `float32`,
      `float64`. Defaults to `float32`.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      `tf.random.set_seed`
      for behavior.
    name: A name for the operation (optional).
  Returns:
    A tensor of the specified shape filled with random normal values.
  """
  with ops.name_scope(name, "random_normal", [shape, mean, stddev]) as name:
    shape_tensor = shape_util.shape_tensor(shape)
    mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops.random_standard_normal(
        shape_tensor, dtype, seed=seed1, seed2=seed2)
    mul = rnd * stddev_tensor
    value = math_ops.add(mul, mean_tensor, name=name)
    shape_util.maybe_set_static_shape(value, shape)
    return value
