@tf_export("distribute.has_strategy")
def has_strategy():
  """Return if there is a current non-default `tf.distribute.Strategy`.
  ```
  assert not tf.distribute.has_strategy()
  with strategy.scope():
    assert tf.distribute.has_strategy()
  ```
  Returns:
    True if inside a `with strategy.scope():`.
  """
  return get_strategy() is not _get_default_strategy()
