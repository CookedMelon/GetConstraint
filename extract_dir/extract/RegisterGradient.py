@tf_export("RegisterGradient")
class RegisterGradient(object):
  """A decorator for registering the gradient function for an op type.
  This decorator is only used when defining a new op type. For an op
  with `m` inputs and `n` outputs, the gradient function is a function
  that takes the original `Operation` and `n` `Tensor` objects
  (representing the gradients with respect to each output of the op),
  and returns `m` `Tensor` objects (representing the partial gradients
  with respect to each input of the op).
  For example, assuming that operations of type `"Sub"` take two
  inputs `x` and `y`, and return a single output `x - y`, the
  following gradient function would be registered:
  ```python
  @tf.RegisterGradient("Sub")
  def _sub_grad(unused_op, grad):
    return grad, tf.negative(grad)
  ```
  The decorator argument `op_type` is the string type of an
  operation. This corresponds to the `OpDef.name` field for the proto
  that defines the operation.
  """
  __slots__ = ["_op_type"]
  def __init__(self, op_type):
    """Creates a new decorator with `op_type` as the Operation type.
    Args:
      op_type: The string type of an operation. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
    Raises:
      TypeError: If `op_type` is not string.
    """
    if not isinstance(op_type, str):
      raise TypeError("op_type must be a string")
    self._op_type = op_type
  def __call__(self, f):
    """Registers the function `f` as gradient function for `op_type`."""
    gradient_registry.register(f, self._op_type)
    return f
