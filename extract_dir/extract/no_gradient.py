@tf_export("no_gradient", v1=["no_gradient", "NotDifferentiable", "NoGradient"])
def no_gradient(op_type):
  """Specifies that ops of type `op_type` is not differentiable.
  This function should *not* be used for operations that have a
  well-defined gradient that is not yet implemented.
  This function is only used when defining a new op type. It may be
  used for ops such as `tf.size()` that are not differentiable.  For
  example:
  ```python
  tf.no_gradient("Size")
  ```
  The gradient computed for 'op_type' will then propagate zeros.
  For ops that have a well-defined gradient but are not yet implemented,
  no declaration should be made, and an error *must* be thrown if
  an attempt to request its gradient is made.
  Args:
    op_type: The string type of an operation. This corresponds to the
      `OpDef.name` field for the proto that defines the operation.
  Raises:
    TypeError: If `op_type` is not a string.
  """
  if not isinstance(op_type, str):
    raise TypeError("op_type must be a string")
  gradient_registry.register(None, op_type)
