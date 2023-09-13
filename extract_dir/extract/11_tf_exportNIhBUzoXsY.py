"/home/cc/Workspace/tfconstraint/python/saved_model/signature_def_utils_impl.py"
@tf_export(
    v1=[
        'saved_model.classification_signature_def',
        'saved_model.signature_def_utils.classification_signature_def'
    ])
@deprecation.deprecated_endpoints(
    'saved_model.signature_def_utils.classification_signature_def')
def classification_signature_def(examples, classes, scores):
  """Creates classification signature from given examples and predictions.
  This function produces signatures intended for use with the TensorFlow Serving
  Classify API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.
  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    classes: A string `Tensor`.  Note that the ClassificationResponse message
      requires that class labels are strings, not integers or anything else.
    scores: a float `Tensor`.
  Returns:
    A classification-flavored signature_def.
  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('Classification `examples` cannot be None.')
  if not isinstance(examples, ops.Tensor):
    raise ValueError('Classification `examples` must be a string Tensor. '
                     f'Found `examples` of type {type(examples)}.')
  if classes is None and scores is None:
    raise ValueError('Classification `classes` and `scores` cannot both be '
                     'None.')
  input_tensor_info = utils.build_tensor_info(examples)
  if input_tensor_info.dtype != types_pb2.DT_STRING:
    raise ValueError('Classification input tensors must be of type string. '
                     f'Found tensors of type {input_tensor_info.dtype}')
  signature_inputs = {signature_constants.CLASSIFY_INPUTS: input_tensor_info}
  signature_outputs = {}
  if classes is not None:
    classes_tensor_info = utils.build_tensor_info(classes)
    if classes_tensor_info.dtype != types_pb2.DT_STRING:
      raise ValueError('Classification classes must be of type string Tensor. '
                       f'Found tensors of type {classes_tensor_info.dtype}.`')
    signature_outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES] = (
        classes_tensor_info)
  if scores is not None:
    scores_tensor_info = utils.build_tensor_info(scores)
    if scores_tensor_info.dtype != types_pb2.DT_FLOAT:
      raise ValueError('Classification scores must be a float Tensor.')
    signature_outputs[signature_constants.CLASSIFY_OUTPUT_SCORES] = (
        scores_tensor_info)
  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.CLASSIFY_METHOD_NAME)
  return signature_def
