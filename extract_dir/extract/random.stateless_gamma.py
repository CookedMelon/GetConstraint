@tf_export("random.stateless_gamma")
@dispatch.add_dispatch_support
def stateless_random_gamma(shape,
                           seed,
                           alpha,
                           beta=None,
                           dtype=dtypes.float32,
                           name=None):
  """Outputs deterministic pseudorandom values from a gamma distribution.
  The generated values follow a gamma distribution with specified concentration
  (`alpha`) and inverse scale (`beta`) parameters.
  This is a stateless version of `tf.random.gamma`: if run twice with the same
  seeds and shapes, it will produce the same pseudorandom numbers. The output is
  consistent across multiple runs on the same hardware (and between CPU and
  GPU),
  but may change between versions of TensorFlow or on non-CPU/GPU hardware.
  A slight difference exists in the interpretation of the `shape` parameter
  between `stateless_gamma` and `gamma`: in `gamma`, the `shape` is always
  prepended to the shape of the broadcast of `alpha` with `beta`; whereas in
  `stateless_gamma` the `shape` parameter must always encompass the shapes of
  each of `alpha` and `beta` (which must broadcast together to match the
  trailing dimensions of `shape`).
  Note: Because internal calculations are done using `float64` and casting has
  `floor` semantics, we must manually map zero outcomes to the smallest
  possible positive floating-point value, i.e., `np.finfo(dtype).tiny`.  This
  means that `np.finfo(dtype).tiny` occurs more frequently than it otherwise
  should.  This bias can only happen for small values of `alpha`, i.e.,
  `alpha << 1` or large values of `beta`, i.e., `beta >> 1`.
  The samples are differentiable w.r.t. alpha and beta.
  The derivatives are computed using the approach described in
  (Figurnov et al., 2018).
  Example:
  ```python
  samples = tf.random.stateless_gamma([10, 2], seed=[12, 34], alpha=[0.5, 1.5])
  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
  # the samples drawn from each distribution
  samples = tf.random.stateless_gamma([7, 5, 2], seed=[12, 34], alpha=[.5, 1.5])
  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
  # represents the 7x5 samples drawn from each of the two distributions
  alpha = tf.constant([[1.], [3.], [5.]])
  beta = tf.constant([[3., 4.]])
  samples = tf.random.stateless_gamma(
      [30, 3, 2], seed=[12, 34], alpha=alpha, beta=beta)
  # samples has shape [30, 3, 2], with 30 samples each of 3x2 distributions.
  with tf.GradientTape() as tape:
    tape.watch([alpha, beta])
    loss = tf.reduce_mean(tf.square(tf.random.stateless_gamma(
        [30, 3, 2], seed=[12, 34], alpha=alpha, beta=beta)))
  dloss_dalpha, dloss_dbeta = tape.gradient(loss, [alpha, beta])
  # unbiased stochastic derivatives of the loss function
  alpha.shape == dloss_dalpha.shape  # True
  beta.shape == dloss_dbeta.shape  # True
  ```
  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    alpha: Tensor. The concentration parameter of the gamma distribution. Must
      be broadcastable with `beta`, and broadcastable with the rightmost
      dimensions of `shape`.
    beta: Tensor. The inverse scale parameter of the gamma distribution. Must be
      broadcastable with `alpha` and broadcastable with the rightmost dimensions
      of `shape`.
    dtype: Floating point dtype of `alpha`, `beta`, and the output.
    name: A name for the operation (optional).
  Returns:
    samples: A Tensor of the specified shape filled with random gamma values.
      For each i, each `samples[..., i] is an independent draw from the gamma
      distribution with concentration alpha[i] and scale beta[i].
  """
  with ops.name_scope(name, "stateless_random_gamma",
                      [shape, seed, alpha, beta]) as name:
    shape = shape_util.shape_tensor(shape)
    alpha = ops.convert_to_tensor(alpha, dtype=dtype, name="alpha")
    beta = ops.convert_to_tensor(
        beta if beta is not None else 1, name="beta", dtype=dtype)
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(alpha), array_ops.shape(beta))
    alpha_broadcast = array_ops.broadcast_to(alpha, broadcast_shape)
    alg = "auto_select"
    key, counter, alg = _get_key_counter_alg(seed, alg)
    rnd = gen_stateless_random_ops_v2.stateless_random_gamma_v3(
        shape, key=key, counter=counter, alg=alg, alpha=alpha_broadcast)
    result = math_ops.maximum(
        np.finfo(alpha.dtype.as_numpy_dtype).tiny, rnd / beta)
    shape_util.maybe_set_static_shape(result, shape)
    return result
