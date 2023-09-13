@tf_export("random.shuffle", v1=["random.shuffle", "random_shuffle"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_shuffle")
def random_shuffle(value, seed=None, name=None):
  """Randomly shuffles a tensor along its first dimension.
  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:
  ```python
  [[1, 2],       [[5, 6],
   [3, 4],  ==>   [1, 2],
   [5, 6]]        [3, 4]]
  ```
  Args:
    value: A Tensor to be shuffled.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      `tf.random.set_seed`
      for behavior.
    name: A name for the operation (optional).
  Returns:
    A tensor of same shape and type as `value`, shuffled along its first
    dimension.
  """
  seed1, seed2 = random_seed.get_seed(seed)
  return gen_random_ops.random_shuffle(
      value, seed=seed1, seed2=seed2, name=name)
