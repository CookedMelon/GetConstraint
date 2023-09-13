@tf_export("distribute.experimental_set_strategy")
def experimental_set_strategy(strategy):
  """Set a `tf.distribute.Strategy` as current without `with strategy.scope()`.
  ```
  tf.distribute.experimental_set_strategy(strategy1)
  f()
  tf.distribute.experimental_set_strategy(strategy2)
  g()
  tf.distribute.experimental_set_strategy(None)
  h()
  ```
  is equivalent to:
  ```
  with strategy1.scope():
    f()
  with strategy2.scope():
    g()
  h()
  ```
  In general, you should use the `with strategy.scope():` API, but this
  alternative may be convenient in notebooks where you would have to put
  each cell in a `with strategy.scope():` block.
  Note: This should only be called outside of any TensorFlow scope to
  avoid improper nesting.
  Args:
    strategy: A `tf.distribute.Strategy` object or None.
  Raises:
    RuntimeError: If called inside a `with strategy.scope():`.
  """
  old_scope = ops.get_default_graph()._global_distribute_strategy_scope  # pylint: disable=protected-access
  if old_scope is not None:
    old_scope.__exit__(None, None, None)
    ops.get_default_graph()._global_distribute_strategy_scope = None  # pylint: disable=protected-access
  if has_strategy():
    raise RuntimeError(
        "Must not be called inside a `tf.distribute.Strategy` scope.")
  if strategy is not None:
    new_scope = strategy.scope()
    new_scope.__enter__()
    ops.get_default_graph()._global_distribute_strategy_scope = new_scope  # pylint: disable=protected-access
