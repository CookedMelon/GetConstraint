@tf_export("is_symbolic_tensor", v1=["is_symbolic_tensor"])
def is_symbolic_tensor(tensor):
  """Test if `tensor` is a symbolic Tensor.
  Args:
    tensor: a tensor-like object
  Returns:
    True if `tensor` is a symbolic tensor (not an eager tensor).
  """
  return type(tensor) == Tensor  # pylint: disable=unidiomatic-typecheck
