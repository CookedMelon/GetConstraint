@tf_export("distribute.get_strategy")
def get_strategy():
  """Returns the current `tf.distribute.Strategy` object.
  Typically only used in a cross-replica context:
  ```
  if tf.distribute.in_cross_replica_context():
    strategy = tf.distribute.get_strategy()
    ...
  ```
  Returns:
    A `tf.distribute.Strategy` object. Inside a `with strategy.scope()` block,
    it returns `strategy`, otherwise it returns the default (single-replica)
    `tf.distribute.Strategy` object.
  """
  return _get_per_thread_mode().strategy
