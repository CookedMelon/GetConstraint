@tf_export("errors.OperatorNotAllowedInGraphError", v1=[])
class OperatorNotAllowedInGraphError(TypeError):
  """Raised when an unsupported operator is present in Graph execution.
  For example, using a `tf.Tensor` as a Python `bool` inside a Graph will
  raise `OperatorNotAllowedInGraphError`. Iterating over values inside a
  `tf.Tensor` is also not supported in Graph execution.
  Example:
  >>> @tf.function
  ... def iterate_over(t):
  ...   a,b,c = t
  ...   return a
  >>>
  >>> iterate_over(tf.constant([1, 2, 3]))
  Traceback (most recent call last):
  ...
  OperatorNotAllowedInGraphError: ...
  """
  pass
