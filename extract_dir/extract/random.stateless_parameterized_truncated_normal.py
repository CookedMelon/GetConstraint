@tf_export("random.stateless_parameterized_truncated_normal")
def stateless_parameterized_truncated_normal(shape,
                                             seed,
                                             means=0.0,
                                             stddevs=1.0,
                                             minvals=-2.0,
                                             maxvals=2.0,
                                             name=None):
  """Outputs random values from a truncated normal distribution.
  The generated values follow a normal distribution with specified mean and
  standard deviation, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.
  Examples:
  Sample from a Truncated normal, with deferring shape parameters that
  broadcast.
  >>> means = 0.
  >>> stddevs = tf.math.exp(tf.random.uniform(shape=[2, 3]))
  >>> minvals = [-1., -2., -1000.]
  >>> maxvals = [[10000.], [1.]]
  >>> y = tf.random.stateless_parameterized_truncated_normal(
  ...   shape=[10, 2, 3], seed=[7, 17],
  ...   means=means, stddevs=stddevs, minvals=minvals, maxvals=maxvals)
  >>> y.shape
  TensorShape([10, 2, 3])
  Args:
    shape: A 1-D integer `Tensor` or Python array. The shape of the output
      tensor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    means: A `Tensor` or Python value of type `dtype`. The mean of the truncated
      normal distribution. This must broadcast with `stddevs`, `minvals` and
      `maxvals`, and the broadcasted shape must be dominated by `shape`.
    stddevs: A `Tensor` or Python value of type `dtype`. The standard deviation
      of the truncated normal distribution. This must broadcast with `means`,
      `minvals` and `maxvals`, and the broadcasted shape must be dominated by
      `shape`.
    minvals: A `Tensor` or Python value of type `dtype`. The minimum value of
      the truncated normal distribution. This must broadcast with `means`,
      `stddevs` and `maxvals`, and the broadcasted shape must be dominated by
      `shape`.
    maxvals: A `Tensor` or Python value of type `dtype`. The maximum value of
      the truncated normal distribution. This must broadcast with `means`,
      `stddevs` and `minvals`, and the broadcasted shape must be dominated by
      `shape`.
    name: A name for the operation (optional).
  Returns:
    A tensor of the specified shape filled with random truncated normal values.
  """
  with ops.name_scope(name, "stateless_parameterized_truncated_normal",
                      [shape, means, stddevs, minvals, maxvals]) as name:
    shape_tensor = shape_util.shape_tensor(shape)
    means_tensor = ops.convert_to_tensor(means, name="means")
    stddevs_tensor = ops.convert_to_tensor(stddevs, name="stddevs")
    minvals_tensor = ops.convert_to_tensor(minvals, name="minvals")
    maxvals_tensor = ops.convert_to_tensor(maxvals, name="maxvals")
    rnd = gen_stateless_random_ops.stateless_parameterized_truncated_normal(
        shape_tensor, seed, means_tensor, stddevs_tensor, minvals_tensor,
        maxvals_tensor)
    shape_util.maybe_set_static_shape(rnd, shape)
    return rnd
