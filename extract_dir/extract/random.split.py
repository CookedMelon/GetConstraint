@tf_export("random.split", "random.experimental.stateless_split")
@dispatch.add_dispatch_support
def split(seed, num=2, alg="auto_select"):
  """Splits an RNG seed into `num` new seeds by adding a leading axis.
  Example:
  >>> seed = [1, 2]
  >>> new_seeds = tf.random.split(seed, num=3)
  >>> print(new_seeds)
  tf.Tensor(
  [[1105988140 1738052849]
   [-335576002  370444179]
   [  10670227 -246211131]], shape=(3, 2), dtype=int32)
  >>> tf.random.stateless_normal(shape=[3], seed=new_seeds[0, :])
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.59835213, -0.9578608 ,
  0.9002807 ], dtype=float32)>
  Args:
    seed: an RNG seed (a tensor with shape [2] and dtype `int32` or
      `int64`). (When using XLA, only `int32` is allowed.)
    num: optional, a positive integer or scalar tensor indicating the number of
      seeds to produce (default 2).
    alg: The RNG algorithm used to generate the random numbers. See
      `tf.random.stateless_uniform` for a detailed explanation.
  Returns:
    A tensor with shape [num, 2] representing `num` new seeds. It will have the
    same dtype as `seed` (if `seed` doesn't have an explict dtype, the dtype
    will be determined by `tf.convert_to_tensor`).
  """
  seed = ops.convert_to_tensor(seed)
  return stateless_random_uniform(shape=[num, 2], seed=seed, dtype=seed.dtype,
                                  minval=None, maxval=None, alg=alg)
