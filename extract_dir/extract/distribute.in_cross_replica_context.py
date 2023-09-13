@tf_export("distribute.in_cross_replica_context")
def in_cross_replica_context():
  """Returns `True` if in a cross-replica context.
  See `tf.distribute.get_replica_context` for details.
  ```
  assert not tf.distribute.in_cross_replica_context()
  with strategy.scope():
    assert tf.distribute.in_cross_replica_context()
    def f():
      assert not tf.distribute.in_cross_replica_context()
    strategy.run(f)
  ```
  Returns:
    `True` if in a cross-replica context (`get_replica_context()` returns
    `None`), or `False` if in a replica context (`get_replica_context()` returns
    non-`None`).
  """
  return _get_per_thread_mode().cross_replica_context is not None
