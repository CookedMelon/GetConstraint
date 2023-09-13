@tf_export("random.stateless_binomial")
@dispatch.add_dispatch_support
def stateless_random_binomial(shape,
                              seed,
                              counts,
                              probs,
                              output_dtype=dtypes.int32,
                              name=None):
  """Outputs deterministic pseudorandom values from a binomial distribution.
  The generated values follow a binomial distribution with specified count and
  probability of success parameters.
  This is a stateless version of `tf.random.Generator.binomial`: if run twice
  with the same seeds and shapes, it will produce the same pseudorandom numbers.
  The output is consistent across multiple runs on the same hardware (and
  between CPU and GPU), but may change between versions of TensorFlow or on
  non-CPU/GPU hardware.
  Example:
  ```python
  counts = [10., 20.]
  # Probability of success.
  probs = [0.8]
  binomial_samples = tf.random.stateless_binomial(
      shape=[2], seed=[123, 456], counts=counts, probs=probs)
  counts = ... # Shape [3, 1, 2]
  probs = ...  # Shape [1, 4, 2]
  shape = [3, 4, 3, 4, 2]
  # Sample shape will be [3, 4, 3, 4, 2]
  binomial_samples = tf.random.stateless_binomial(
      shape=shape, seed=[123, 456], counts=counts, probs=probs)
  ```
  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    counts: Tensor. The counts of the binomial distribution. Must be
      broadcastable with `probs`, and broadcastable with the rightmost
      dimensions of `shape`.
    probs: Tensor. The probability of success for the binomial distribution.
      Must be broadcastable with `counts` and broadcastable with the rightmost
      dimensions of `shape`.
    output_dtype: The type of the output. Default: tf.int32
    name: A name for the operation (optional).
  Returns:
    samples: A Tensor of the specified shape filled with random binomial
      values.  For each i, each samples[..., i] is an independent draw from
      the binomial distribution on counts[i] trials with probability of
      success probs[i].
  """
  with ops.name_scope(name, "stateless_random_binomial",
                      [shape, seed, counts, probs]) as name:
    shape = shape_util.shape_tensor(shape)
    probs = ops.convert_to_tensor(
        probs, dtype_hint=dtypes.float32, name="probs")
    counts = ops.convert_to_tensor(
        counts, dtype_hint=probs.dtype, name="counts")
    result = gen_stateless_random_ops.stateless_random_binomial(
        shape=shape, seed=seed, counts=counts, probs=probs, dtype=output_dtype)
    shape_util.maybe_set_static_shape(result, shape)
    return result
