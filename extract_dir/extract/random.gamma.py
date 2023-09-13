@tf_export("random.gamma", v1=["random.gamma", "random_gamma"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_gamma")
def random_gamma(shape,
                 alpha,
                 beta=None,
                 dtype=dtypes.float32,
                 seed=None,
                 name=None):
  """Draws `shape` samples from each of the given Gamma distribution(s).
  `alpha` is the shape parameter describing the distribution(s), and `beta` is
  the inverse scale parameter(s).
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
  samples = tf.random.gamma([10], [0.5, 1.5])
  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
  # the samples drawn from each distribution
  samples = tf.random.gamma([7, 5], [0.5, 1.5])
  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
  # represents the 7x5 samples drawn from each of the two distributions
  alpha = tf.constant([[1.],[3.],[5.]])
  beta = tf.constant([[3., 4.]])
  samples = tf.random.gamma([30], alpha=alpha, beta=beta)
  # samples has shape [30, 3, 2], with 30 samples each of 3x2 distributions.
  loss = tf.reduce_mean(tf.square(samples))
  dloss_dalpha, dloss_dbeta = tf.gradients(loss, [alpha, beta])
  # unbiased stochastic derivatives of the loss function
  alpha.shape == dloss_dalpha.shape  # True
  beta.shape == dloss_dbeta.shape  # True
  ```
  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output samples
      to be drawn per alpha/beta-parameterized distribution.
    alpha: A Tensor or Python value or N-D array of type `dtype`. `alpha`
      provides the shape parameter(s) describing the gamma distribution(s) to
      sample. Must be broadcastable with `beta`.
    beta: A Tensor or Python value or N-D array of type `dtype`. Defaults to 1.
      `beta` provides the inverse scale parameter(s) of the gamma
      distribution(s) to sample. Must be broadcastable with `alpha`.
    dtype: The type of alpha, beta, and the output: `float16`, `float32`, or
      `float64`.
    seed: A Python integer. Used to create a random seed for the distributions.
      See
      `tf.random.set_seed`
      for behavior.
    name: Optional name for the operation.
  Returns:
    samples: a `Tensor` of shape
      `tf.concat([shape, tf.shape(alpha + beta)], axis=0)` with values of type
      `dtype`.
  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """
  with ops.name_scope(name, "random_gamma", [shape, alpha, beta]):
    shape = ops.convert_to_tensor(shape, name="shape", dtype=dtypes.int32)
    alpha = ops.convert_to_tensor(alpha, name="alpha", dtype=dtype)
    beta = ops.convert_to_tensor(
        beta if beta is not None else 1, name="beta", dtype=dtype)
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(alpha), array_ops.shape(beta))
    alpha_broadcast = array_ops.broadcast_to(alpha, broadcast_shape)
    seed1, seed2 = random_seed.get_seed(seed)
    result = math_ops.maximum(
        np.finfo(alpha.dtype.as_numpy_dtype).tiny,
        gen_random_ops.random_gamma(
            shape, alpha_broadcast, seed=seed1, seed2=seed2) / beta)
    _maybe_set_static_shape_helper(result, shape, alpha_broadcast)
    return result
