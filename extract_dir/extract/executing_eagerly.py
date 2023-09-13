@tf_export("executing_eagerly", v1=[])
def executing_eagerly():
  """Checks whether the current thread has eager execution enabled.
  Eager execution is enabled by default and this API returns `True`
  in most of cases. However, this API might return `False` in the following use
  cases.
  *  Executing inside `tf.function`, unless under `tf.init_scope` or
     `tf.config.run_functions_eagerly(True)` is previously called.
  *  Executing inside a transformation function for `tf.dataset`.
  *  `tf.compat.v1.disable_eager_execution()` is called.
  General case:
  >>> print(tf.executing_eagerly())
  True
  Inside `tf.function`:
  >>> @tf.function
  ... def fn():
  ...   with tf.init_scope():
  ...     print(tf.executing_eagerly())
  ...   print(tf.executing_eagerly())
  >>> fn()
  True
  False
  Inside `tf.function` after `tf.config.run_functions_eagerly(True)` is called:
  >>> tf.config.run_functions_eagerly(True)
  >>> @tf.function
  ... def fn():
  ...   with tf.init_scope():
  ...     print(tf.executing_eagerly())
  ...   print(tf.executing_eagerly())
  >>> fn()
  True
  True
  >>> tf.config.run_functions_eagerly(False)
  Inside a transformation function for `tf.dataset`:
  >>> def data_fn(x):
  ...   print(tf.executing_eagerly())
  ...   return x
  >>> dataset = tf.data.Dataset.range(100)
  >>> dataset = dataset.map(data_fn)
  False
  Returns:
    `True` if the current thread has eager execution enabled.
  """
  ctx = context_safe()
  if ctx is None:
    return default_execution_mode == EAGER_MODE
  return ctx.executing_eagerly()
