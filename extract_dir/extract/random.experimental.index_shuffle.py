@tf_export("random.experimental.index_shuffle")
@dispatch.add_dispatch_support
def index_shuffle(index, seed, max_index):
  """Outputs the position of `index` in a permutation of `[0, ..., max_index]`.
  For each possible `seed` and `max_index` there is one pseudorandom
  permutation of the sequence `S=[0, ..., max_index]`. Instead of
  materializing the full array we can compute the new position of any
  integer `i` (`0 <= i <= max_index`) in `S`. This can be useful for
  very large `max_index`s by avoiding allocating large chunks of
  memory.
  In the simplest case, `index` and `max_index` are scalars, and
  `seed` is a length-2 vector (as typical for stateless RNGs). But
  you can add a leading batch dimension to all of them. If some of
  them don't have the batch dimension while others do, `index_shuffle`
  will add a batch dimension to the former by broadcasting.
  The input `index` and output can be used as indices to shuffle a
  vector.  For example:
  >>> vector = tf.constant(['e0', 'e1', 'e2', 'e3'])
  >>> indices = tf.random.experimental.index_shuffle(
  ...   index=tf.range(4), seed=[5, 9], max_index=3)
  >>> print(indices)
  tf.Tensor([2 0 1 3], shape=(4,), dtype=int32)
  >>> shuffled_vector = tf.gather(vector, indices)
  >>> print(shuffled_vector)
  tf.Tensor([b'e2' b'e0' b'e1' b'e3'], shape=(4,), dtype=string)
  More usefully, it can be used in a streaming (aka online) scenario such as
  `tf.data`, where each element of `vector` is processed individually and the
  whole `vector` is never materialized in memory.
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset = dataset.map(
  ...  lambda idx: tf.random.experimental.index_shuffle(idx, [5, 8], 9))
  >>> print(list(dataset.as_numpy_iterator()))
  [3, 8, 0, 1, 2, 7, 6, 9, 4, 5]
  This operation is stateless (like the `tf.random.stateless_*`
  functions), meaning the output is fully determined by the `seed`
  (other inputs being equal).  Each `seed` choice corresponds to one
  permutation, so when calling this function multiple times for the
  same shuffling, please make sure to use the same `seed`. For
  example:
  >>> seed = [5, 9]
  >>> idx0 = tf.random.experimental.index_shuffle(0, seed, 3)
  >>> idx1 = tf.random.experimental.index_shuffle(1, seed, 3)
  >>> idx2 = tf.random.experimental.index_shuffle(2, seed, 3)
  >>> idx3 = tf.random.experimental.index_shuffle(3, seed, 3)
  >>> shuffled_vector = tf.gather(vector, [idx0, idx1, idx2, idx3])
  >>> print(shuffled_vector)
  tf.Tensor([b'e2' b'e0' b'e1' b'e3'], shape=(4,), dtype=string)
  Args:
    index: An integer scalar tensor or vector with values in `[0,
      max_index]`.  It can be seen as either a value `v` in the
      sequence `S=[0, ..., max_index]` to be permutated, or as an
      index of an element `e` in a shuffled vector.
    seed: A tensor of shape [2] or [n, 2] with dtype `int32`,
      `uint32`, `int64` or `uint64`.  The RNG seed. If the rank is
      unknown during graph-building time it must be 1 at runtime.
    max_index: A non-negative tensor with the same shape and dtype as
      `index`.  The upper bound (inclusive).
  Returns:
    If all inputs were scalar (shape [2] for `seed`), the output will
    be a scalar with the same dtype as `index`. The output can be seen
    as the new position of `v` in `S`, or as the index of `e` in the
    vector before shuffling.  If one or multiple inputs were vectors
    (shape [n, 2] for `seed`), then the output will be a vector of the
    same size which each element shuffled independently. Scalar values
    are broadcasted in this case.
  """
  # We expect users to pass a seed with shape [2] to be consistent with other
  # stateless_* ops, but the raw op expects shape [3].
  seed = ops.convert_to_tensor(seed)
  # Pad the first dimension with an arbitrary number since our raw op expects
  # shape [3].
  if seed.shape.rank is None:
    paddings = [[1, 0]]
  else:
    paddings = [[1, 0]] + (seed.shape.rank - 1) * [[0, 0]]
  seed = array_ops.pad(seed, paddings, constant_values=498247692)
  return gen_random_index_shuffle_ops.random_index_shuffle(
      index, seed=seed, max_index=max_index, rounds=4)
