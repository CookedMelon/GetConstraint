@tf_export("guarantee_const")
@deprecation.deprecated(None, "Not for public use.")
def guarantee_const(input, name=None):    # pylint: disable=redefined-builtin
  """Promise to the TF runtime that the input tensor is a constant.
  The runtime is then free to make optimizations based on this.
  Returns the input tensor without modification.
  Args:
    input: A `Tensor`.
    name: A name for this operation.
  Returns:
    A `Tensor`. Has the same dtype as `input`.
  """
  return gen_array_ops.guarantee_const(input=input, name=name)
