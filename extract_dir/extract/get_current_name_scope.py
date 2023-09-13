@tf_export("get_current_name_scope", v1=[])
def get_current_name_scope():
  """Returns current full name scope specified by `tf.name_scope(...)`s.
  For example,
  ```python
  with tf.name_scope("outer"):
    tf.get_current_name_scope()  # "outer"
    with tf.name_scope("inner"):
      tf.get_current_name_scope()  # "outer/inner"
  ```
  In other words, `tf.get_current_name_scope()` returns the op name prefix that
  will be prepended to, if an op is created at that place.
  Note that `@tf.function` resets the name scope stack as shown below.
  ```
  with tf.name_scope("outer"):
    @tf.function
    def foo(x):
      with tf.name_scope("inner"):
        return tf.add(x * x)  # Op name is "inner/Add", not "outer/inner/Add"
  ```
  """
  ctx = context.context()
  if ctx.executing_eagerly():
    return ctx.scope_name.rstrip("/")
  else:
    return get_default_graph().get_name_scope()
