"/home/cc/Workspace/tfconstraint/python/saved_model/signature_def_utils_impl.py"
@tf_export(
    v1=[
        'saved_model.regression_signature_def',
        'saved_model.signature_def_utils.regression_signature_def'
    ])
@deprecation.deprecated_endpoints(
    'saved_model.signature_def_utils.regression_signature_def')
def regression_signature_def(examples, predictions):
  """Creates regression signature from given examples and predictions.
  This function produces signatures intended for use with the TensorFlow Serving
  Regress API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.
  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    predictions: A float `Tensor`.
  Returns:
    A regression-flavored signature_def.
  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('Regression `examples` cannot be None.')
  if not isinstance(examples, ops.Tensor):
    raise ValueError('Expected regression `examples` to be of type Tensor. '
                     f'Found `examples` of type {type(examples)}.')
  if predictions is None:
    raise ValueError('Regression `predictions` cannot be None.')
  input_tensor_info = utils.build_tensor_info(examples)
  if input_tensor_info.dtype != types_pb2.DT_STRING:
    raise ValueError('Regression input tensors must be of type string. '
                     f'Found tensors with type {input_tensor_info.dtype}.')
  signature_inputs = {signature_constants.REGRESS_INPUTS: input_tensor_info}
  output_tensor_info = utils.build_tensor_info(predictions)
  if output_tensor_info.dtype != types_pb2.DT_FLOAT:
    raise ValueError('Regression output tensors must be of type float. '
                     f'Found tensors with type {output_tensor_info.dtype}.')
  signature_outputs = {signature_constants.REGRESS_OUTPUTS: output_tensor_info}
  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.REGRESS_METHOD_NAME)
  return signature_def
