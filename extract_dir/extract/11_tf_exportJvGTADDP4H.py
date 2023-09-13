"/home/cc/Workspace/tfconstraint/python/saved_model/signature_def_utils_impl.py"
@tf_export(
    v1=[
        'saved_model.is_valid_signature',
        'saved_model.signature_def_utils.is_valid_signature'
    ])
@deprecation.deprecated_endpoints(
    'saved_model.signature_def_utils.is_valid_signature')
def is_valid_signature(signature_def):
  """Determine whether a SignatureDef can be served by TensorFlow Serving."""
  if signature_def is None:
    return False
  return (_is_valid_classification_signature(signature_def) or
          _is_valid_regression_signature(signature_def) or
          _is_valid_predict_signature(signature_def))
def _is_valid_predict_signature(signature_def):
  """Determine whether the argument is a servable 'predict' SignatureDef."""
  if signature_def.method_name != signature_constants.PREDICT_METHOD_NAME:
    return False
  if not signature_def.inputs.keys():
    return False
  if not signature_def.outputs.keys():
    return False
  return True
def _is_valid_regression_signature(signature_def):
  """Determine whether the argument is a servable 'regress' SignatureDef."""
  if signature_def.method_name != signature_constants.REGRESS_METHOD_NAME:
    return False
  if (set(signature_def.inputs.keys())
      != set([signature_constants.REGRESS_INPUTS])):
    return False
  if (signature_def.inputs[signature_constants.REGRESS_INPUTS].dtype !=
      types_pb2.DT_STRING):
    return False
  if (set(signature_def.outputs.keys())
      != set([signature_constants.REGRESS_OUTPUTS])):
    return False
  if (signature_def.outputs[signature_constants.REGRESS_OUTPUTS].dtype !=
      types_pb2.DT_FLOAT):
    return False
  return True
def _is_valid_classification_signature(signature_def):
  """Determine whether the argument is a servable 'classify' SignatureDef."""
  if signature_def.method_name != signature_constants.CLASSIFY_METHOD_NAME:
    return False
  if (set(signature_def.inputs.keys())
      != set([signature_constants.CLASSIFY_INPUTS])):
    return False
  if (signature_def.inputs[signature_constants.CLASSIFY_INPUTS].dtype !=
      types_pb2.DT_STRING):
    return False
  allowed_outputs = set([signature_constants.CLASSIFY_OUTPUT_CLASSES,
                         signature_constants.CLASSIFY_OUTPUT_SCORES])
  if not signature_def.outputs.keys():
    return False
  if set(signature_def.outputs.keys()) - allowed_outputs:
    return False
  if (signature_constants.CLASSIFY_OUTPUT_CLASSES in signature_def.outputs
      and
      signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES].dtype
      != types_pb2.DT_STRING):
    return False
  if (signature_constants.CLASSIFY_OUTPUT_SCORES in signature_def.outputs
      and
      signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_SCORES].dtype !=
      types_pb2.DT_FLOAT):
    return False
  return True
def op_signature_def(op, key):
  """Creates a signature def with the output pointing to an op.
  Note that op isn't strictly enforced to be an Op object, and may be a Tensor.
  It is recommended to use the build_signature_def() function for Tensors.
  Args:
    op: An Op (or possibly Tensor).
    key: Key to graph element in the SignatureDef outputs.
  Returns:
    A SignatureDef with a single output pointing to the op.
  """
  # Use build_tensor_info_from_op, which creates a TensorInfo from the element's
  # name.
  return build_signature_def(outputs={key: utils.build_tensor_info_from_op(op)})
def load_op_from_signature_def(signature_def, key, import_scope=None):
  """Load an Op from a SignatureDef created by op_signature_def().
  Args:
    signature_def: a SignatureDef proto
    key: string key to op in the SignatureDef outputs.
    import_scope: Scope used to import the op
  Returns:
    Op (or possibly Tensor) in the graph with the same name as saved in the
      SignatureDef.
  Raises:
    NotFoundError: If the op could not be found in the graph.
  """
  tensor_info = signature_def.outputs[key]
  try:
    # The init and train ops are not strictly enforced to be operations, so
    # retrieve any graph element (can be either op or tensor).
    return utils.get_element_from_tensor_info(
        tensor_info, import_scope=import_scope)
  except KeyError:
    raise errors.NotFoundError(
        None, None,
        f'The key "{key}" could not be found in the graph. Please make sure the'
        ' SavedModel was created by the internal _SavedModelBuilder. If you '
        'are using the public API, please make sure the SignatureDef in the '
        f'SavedModel does not contain the key "{key}".')
