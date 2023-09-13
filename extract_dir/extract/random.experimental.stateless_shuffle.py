@tf_export("random.experimental.stateless_shuffle")
@dispatch.add_dispatch_support
def stateless_shuffle(value, seed, alg="auto_select", name=None):
  """Randomly and deterministically shuffles a tensor along its first dimension.
  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:
  ```python
  [[1, 2],       [[5, 6],
   [3, 4],  ==>   [1, 2],
   [5, 6]]        [3, 4]]
  ```
  >>> v = tf.constant([[1, 2], [3, 4], [5, 6]])
  >>> shuffled = tf.random.experimental.stateless_shuffle(v, seed=[8, 9])
  >>> print(shuffled)
  tf.Tensor(
  [[5 6]
    [1 2]
    [3 4]], shape=(3, 2), dtype=int32)
  This is a stateless version of `tf.random.shuffle`: if run twice with the
  same `value` and `seed`, it will produce the same result.  The
  output is consistent across multiple runs on the same hardware (and between
  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.
  Args:
    value: A Tensor to be shuffled.
    seed: A shape [2] Tensor. The seed to the random number generator. Must have
      dtype `int32` or `int64`.
    alg: The RNG algorithm used to generate the random numbers. See
      `tf.random.stateless_uniform` for a detailed explanation.
    name: A name for the operation.
  Returns:
    A tensor of same shape and type as `value`, shuffled along its first
    dimension.
  """
  with ops.name_scope(name, "stateless_shuffle", [value, seed]) as name:
    key, counter, alg = _get_key_counter_alg(seed, alg)
    return gen_stateless_random_ops_v2.stateless_shuffle(
        value, key=key, counter=counter, alg=alg)
