@tf_export("stop_gradient")
@dispatch.add_dispatch_support
def stop_gradient(input, name=None):  # pylint: disable=redefined-builtin
  """Stops gradient computation.
  NOTE: This docstring is patched out below. See
  tensorflow/core/api_def/base_api/api_def_StopGradient.pbtxt for the full
  docstring. That file determines the public documentation page.
  Args:
    input: A `Tensor`.
    name: A name for this operation.
  Returns:
    A `Tensor`. Has the same dtype as `input`.
  """
  # Don't expand ResourceVariables, so stop_gradient(variable) will return a
  # Tensor.
  if (isinstance(input, composite_tensor.CompositeTensor) and
      not _pywrap_utils.IsResourceVariable(input)):
    return nest.map_structure(stop_gradient, input, expand_composites=True)
  # The StopGradient op has a gradient function registered which returns None
  # (meaning statically known to be zero). For correctness, that's all we
  # need. However, tf.GradientTape often makes decisions about what to keep in
  # memory based on which forward-pass tensors are currently being watched, and
  # returning None in a gradient is not sufficient to stop watching a tensor
  # since the backward function doesn't run in the forward pass. Pausing the
  # tape around this op instructs any tf.GradientTapes to ignore the
  # forward-pass output of StopGradient, which may be much more efficient.
  with record.stop_recording():
    return gen_array_ops.stop_gradient(input, name=name)
