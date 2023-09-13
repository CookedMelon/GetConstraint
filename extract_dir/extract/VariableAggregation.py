@tf_export("VariableAggregation", v1=[])
class VariableAggregationV2(enum.Enum):
  """Indicates how a distributed variable will be aggregated.
  `tf.distribute.Strategy` distributes a model by making multiple copies
  (called "replicas") acting on different elements of the input batch in a
  data parallel model. When performing some variable-update operation,
  for example `var.assign_add(x)`, in a model, we need to resolve how to combine
  the different values for `x` computed in the different replicas.
  * `NONE`: This is the default, giving an error if you use a
    variable-update operation with multiple replicas.
  * `SUM`: Add the updates across replicas.
  * `MEAN`: Take the arithmetic mean ("average") of the updates across replicas.
  * `ONLY_FIRST_REPLICA`: This is for when every replica is performing the same
    update, but we only want to perform the update once. Used, e.g., for the
    global step counter.
  For example:
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> with strategy.scope():
  ...   v = tf.Variable(5.0, aggregation=tf.VariableAggregation.MEAN)
  >>> @tf.function
  ... def update_fn():
  ...   return v.assign_add(1.0)
  >>> strategy.run(update_fn)
  PerReplica:{
    0: <tf.Tensor: shape=(), dtype=float32, numpy=6.0>,
    1: <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
  }
  """
  NONE = 0
  SUM = 1
  MEAN = 2
  ONLY_FIRST_REPLICA = 3
  def __hash__(self):
    return hash(self.value)
  def __eq__(self, other):
    if self is other:
      return True
    elif isinstance(other, VariableAggregation):
      return int(self.value) == int(other.value)
    else:
      return False
