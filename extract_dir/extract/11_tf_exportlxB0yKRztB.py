"/home/cc/Workspace/tfconstraint/python/saved_model/signature_def_utils_impl.py"
@tf_export(
    v1=[
        'saved_model.predict_signature_def',
        'saved_model.signature_def_utils.predict_signature_def'
    ])
@deprecation.deprecated_endpoints(
    'saved_model.signature_def_utils.predict_signature_def')
def predict_signature_def(inputs, outputs):
  """Creates prediction signature from given inputs and outputs.
  This function produces signatures intended for use with the TensorFlow Serving
  Predict API (tensorflow_serving/apis/prediction_service.proto). This API
  imposes no constraints on the input and output types.
  Args:
    inputs: dict of string to `Tensor`.
    outputs: dict of string to `Tensor`.
  Returns:
    A prediction-flavored signature_def.
  Raises:
    ValueError: If inputs or outputs is `None`.
  """
  if inputs is None or not inputs:
    raise ValueError('Prediction `inputs` cannot be None or empty.')
  if outputs is None or not outputs:
    raise ValueError('Prediction `outputs` cannot be None or empty.')
  signature_inputs = {key: utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}
  signature_outputs = {key: utils.build_tensor_info(tensor)
                       for key, tensor in outputs.items()}
  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)
  return signature_def
# LINT.IfChange
def supervised_train_signature_def(
    inputs, loss, predictions=None, metrics=None):
  return _supervised_signature_def(
      signature_constants.SUPERVISED_TRAIN_METHOD_NAME, inputs, loss=loss,
      predictions=predictions, metrics=metrics)
def supervised_eval_signature_def(
    inputs, loss, predictions=None, metrics=None):
  return _supervised_signature_def(
      signature_constants.SUPERVISED_EVAL_METHOD_NAME, inputs, loss=loss,
      predictions=predictions, metrics=metrics)
def _supervised_signature_def(
    method_name, inputs, loss=None, predictions=None,
    metrics=None):
  """Creates a signature for training and eval data.
  This function produces signatures that describe the inputs and outputs
  of a supervised process, such as training or evaluation, that
  results in loss, metrics, and the like. Note that this function only requires
  inputs to be not None.
  Args:
    method_name: Method name of the SignatureDef as a string.
    inputs: dict of string to `Tensor`.
    loss: dict of string to `Tensor` representing computed loss.
    predictions: dict of string to `Tensor` representing the output predictions.
    metrics: dict of string to `Tensor` representing metric ops.
  Returns:
    A train- or eval-flavored signature_def.
  Raises:
    ValueError: If inputs or outputs is `None`.
  """
  if inputs is None or not inputs:
    raise ValueError(f'{method_name} `inputs` cannot be None or empty.')
  signature_inputs = {key: utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}
  signature_outputs = {}
  for output_set in (loss, predictions, metrics):
    if output_set is not None:
      sig_out = {key: utils.build_tensor_info(tensor)
                 for key, tensor in output_set.items()}
      signature_outputs.update(sig_out)
  signature_def = build_signature_def(
      signature_inputs, signature_outputs, method_name)
  return signature_def
# LINT.ThenChange(//keras/saving/utils_v1/signature_def_utils.py)
