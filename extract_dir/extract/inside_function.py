@tf_export("inside_function", v1=[])
def inside_function():
  """Indicates whether the caller code is executing inside a `tf.function`.
  Returns:
    Boolean, True if the caller code is executing inside a `tf.function`
    rather than eagerly.
  Example:
  >>> tf.inside_function()
  False
  >>> @tf.function
  ... def f():
  ...   print(tf.inside_function())
  >>> f()
  True
  """
  return get_default_graph().building_function
