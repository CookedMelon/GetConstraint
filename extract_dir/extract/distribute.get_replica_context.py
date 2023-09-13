@tf_export("distribute.get_replica_context")
def get_replica_context():
  """Returns the current `tf.distribute.ReplicaContext` or `None`.
  Returns `None` if in a cross-replica context.
  Note that execution:
  1. starts in the default (single-replica) replica context (this function
     will return the default `ReplicaContext` object);
  2. switches to cross-replica context (in which case this will return
     `None`) when entering a `with tf.distribute.Strategy.scope():` block;
  3. switches to a (non-default) replica context inside `strategy.run(fn, ...)`;
  4. if `fn` calls `get_replica_context().merge_call(merge_fn, ...)`, then
     inside `merge_fn` you are back in the cross-replica context (and again
     this function will return `None`).
  Most `tf.distribute.Strategy` methods may only be executed in
  a cross-replica context, in a replica context you should use the
  API of the `tf.distribute.ReplicaContext` object returned by this
  method instead.
  ```
  assert tf.distribute.get_replica_context() is not None  # default
  with strategy.scope():
    assert tf.distribute.get_replica_context() is None
    def f():
      replica_context = tf.distribute.get_replica_context()  # for strategy
      assert replica_context is not None
      tf.print("Replica id: ", replica_context.replica_id_in_sync_group,
               " of ", replica_context.num_replicas_in_sync)
    strategy.run(f)
  ```
  Returns:
    The current `tf.distribute.ReplicaContext` object when in a replica context
    scope, else `None`.
    Within a particular block, exactly one of these two things will be true:
    * `get_replica_context()` returns non-`None`, or
    * `tf.distribute.is_cross_replica_context()` returns True.
  """
  return _get_per_thread_mode().replica_context
