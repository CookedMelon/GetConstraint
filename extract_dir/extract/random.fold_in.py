@tf_export("random.fold_in", "random.experimental.stateless_fold_in")
@dispatch.add_dispatch_support
def fold_in(seed, data, alg="auto_select"):
  """Folds in data to an RNG seed to form a new RNG seed.
  For example, in a distributed-training setting, suppose we have a master seed
  and a replica ID. We want to fold the replica ID into the master seed to
  form a "replica seed" to be used by that replica later on, so that different
  replicas will generate different random numbers but the reproducibility of the
  whole system can still be controlled by the master seed:
  >>> master_seed = [1, 2]
  >>> replica_id = 3
  >>> replica_seed = tf.random.experimental.stateless_fold_in(
  ...   master_seed, replica_id)
  >>> print(replica_seed)
  tf.Tensor([1105988140          3], shape=(2,), dtype=int32)
  >>> tf.random.stateless_normal(shape=[3], seed=replica_seed)
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.03197195, 0.8979765 ,
  0.13253039], dtype=float32)>
  Args:
    seed: an RNG seed (a tensor with shape [2] and dtype `int32` or
      `int64`). (When using XLA, only `int32` is allowed.)
    data: an `int32` or `int64` scalar representing data to be folded in to the
      seed.
    alg: The RNG algorithm used to generate the random numbers. See
      `tf.random.stateless_uniform` for a detailed explanation.
  Returns:
    A new RNG seed that is a deterministic function of the inputs and is
    statistically safe for producing a stream of new pseudo-random values. It
    will have the same dtype as `data` (if `data` doesn't have an explict dtype,
    the dtype will be determined by `tf.convert_to_tensor`).
  """
  data = ops.convert_to_tensor(data)
  seed1 = stateless_random_uniform(shape=[], seed=seed, dtype=data.dtype,
                                   minval=None, maxval=None, alg=alg)
  return array_ops_stack.stack([seed1, data])
