@tf_export("make_tensor_proto")
def make_tensor_proto(values, dtype=None, shape=None, verify_shape=False,
                      allow_broadcast=False):
  """Create a TensorProto.
  In TensorFlow 2.0, representing tensors as protos should no longer be a
  common workflow. That said, this utility function is still useful for
  generating TF Serving request protos:
  ```python
    request = tensorflow_serving.apis.predict_pb2.PredictRequest()
    request.model_spec.name = "my_model"
    request.model_spec.signature_name = "serving_default"
    request.inputs["images"].CopyFrom(tf.make_tensor_proto(X_new))
  ```
  `make_tensor_proto` accepts "values" of a python scalar, a python list, a
  numpy ndarray, or a numpy scalar.
  If "values" is a python scalar or a python list, make_tensor_proto
  first convert it to numpy ndarray. If dtype is None, the
  conversion tries its best to infer the right numpy data
  type. Otherwise, the resulting numpy array has a compatible data
  type with the given dtype.
  In either case above, the numpy ndarray (either the caller provided
  or the auto-converted) must have the compatible type with dtype.
  `make_tensor_proto` then converts the numpy array to a tensor proto.
  If "shape" is None, the resulting tensor proto represents the numpy
  array precisely.
  Otherwise, "shape" specifies the tensor's shape and the numpy array
  can not have more elements than what "shape" specifies.
  Args:
    values:         Values to put in the TensorProto.
    dtype:          Optional tensor_pb2 DataType value.
    shape:          List of integers representing the dimensions of tensor.
    verify_shape:   Boolean that enables verification of a shape of values.
    allow_broadcast:  Boolean that enables allowing scalars and 1 length vector
        broadcasting. Cannot be true when verify_shape is true.
  Returns:
    A `TensorProto`. Depending on the type, it may contain data in the
    "tensor_content" attribute, which is not directly useful to Python programs.
    To access the values you should convert the proto back to a numpy ndarray
    with `tf.make_ndarray(proto)`.
    If `values` is a `TensorProto`, it is immediately returned; `dtype` and
    `shape` are ignored.
  Raises:
    TypeError:  if unsupported types are provided.
    ValueError: if arguments have inappropriate values or if verify_shape is
     True and shape of values is not equals to a shape from the argument.
  """
  if allow_broadcast and verify_shape:
    raise ValueError("allow_broadcast and verify_shape are not both allowed.")
  if isinstance(values, tensor_pb2.TensorProto):
    return values
  if dtype:
    dtype = dtypes.as_dtype(dtype)
  is_quantized = (
      dtype in [
          dtypes.qint8, dtypes.quint8, dtypes.qint16, dtypes.quint16,
          dtypes.qint32
      ])
  if _is_array_like(values):
    values = np.asarray(values)
  # We first convert value to a numpy array or scalar.
  if isinstance(values, (np.ndarray, np.generic)):
    if dtype and dtype.is_numpy_compatible:
      nparray = values.astype(dtype.as_numpy_dtype)
    else:
      nparray = values
  else:
    if values is None:
      raise ValueError("None values not supported.")
    # if dtype is provided, forces numpy array to be the type
    # provided if possible.
    if dtype and dtype.is_numpy_compatible:
      np_dt = dtype.as_numpy_dtype
    else:
      np_dt = None
    # If shape is None, numpy.prod returns None when dtype is not set, but
    # raises exception when dtype is set to np.int64
    if shape is not None and np.prod(shape, dtype=np.int64) == 0:
      nparray = np.empty(shape, dtype=np_dt)
    else:
      _AssertCompatible(values, dtype)
      nparray = np.array(values, dtype=np_dt)
      # check to them.
      # We need to pass in quantized values as tuples, so don't apply the shape
      if (list(nparray.shape) != _GetDenseDimensions(values) and
          not is_quantized):
        raise ValueError(f"Expected values {values} to be a dense tensor with "
                         f"shape {_GetDenseDimensions(values)}, but got shape "
                         f"{list(nparray.shape)}.")
    # python/numpy default float type is float64. We prefer float32 instead.
    if (nparray.dtype == np.float64) and dtype is None:
      nparray = nparray.astype(np.float32)
    # python/numpy default int type is int64. We prefer int32 instead.
    elif (nparray.dtype == np.int64) and dtype is None:
      downcasted_array = nparray.astype(np.int32)
      # Do not down cast if it leads to precision loss.
      if np.array_equal(downcasted_array, nparray):
        nparray = downcasted_array
  # if dtype is provided, it must be compatible with what numpy
  # conversion says.
  numpy_dtype = dtypes.as_dtype(nparray.dtype)
  if numpy_dtype is None:
    raise TypeError(f"Unrecognized data type: {nparray.dtype}.")
  # If dtype was specified and is a quantized type, we convert
  # numpy_dtype back into the quantized version.
  if is_quantized:
    numpy_dtype = dtype
  if dtype is not None and (not hasattr(dtype, "base_dtype") or
                            dtype.base_dtype != numpy_dtype.base_dtype):
    raise TypeError(f"`dtype` {dtype} is not compatible with {values} of "
                    f"dtype {nparray.dtype}.")
  # If shape is not given, get the shape from the numpy array.
  if shape is None:
    shape = nparray.shape
    is_same_size = True
    shape_size = nparray.size
  else:
    shape = [int(dim) for dim in shape]
    shape_size = np.prod(shape, dtype=np.int64)
    is_same_size = shape_size == nparray.size
    if allow_broadcast:
      if nparray.shape == (1,) or nparray.shape == tuple():
        pass
      elif nparray.size != shape_size:
        raise TypeError(f"Expected Tensor's shape: {tuple(shape)}, but got "
                        f"{nparray.shape}.")
    else:
      if verify_shape and nparray.shape != tuple(shape):
        raise TypeError(f"Expected Tensor's shape: {tuple(shape)}, but got "
                        f"{nparray.shape}.")
      if nparray.size > shape_size:
        raise ValueError("Too many elements provided. Takes at most "
                         f"{shape_size:d}, but got {nparray.size:d}.")
  tensor_proto = tensor_pb2.TensorProto(
      dtype=numpy_dtype.as_datatype_enum,
      tensor_shape=tensor_shape.as_shape(shape).as_proto())
  if is_same_size and numpy_dtype in _TENSOR_CONTENT_TYPES and shape_size > 1:
    if nparray.size * nparray.itemsize >= (1 << 31):
      raise ValueError(
          "Cannot create a tensor proto whose content is larger than 2GB.")
    tensor_proto.tensor_content = nparray.tobytes()
    return tensor_proto
  # If we were not given values as a numpy array, compute the proto_values
  # from the given values directly, to avoid numpy trimming nulls from the
  # strings. Since values could be a list of strings, or a multi-dimensional
  # list of lists that might or might not correspond to the given shape,
  # we flatten it conservatively.
  if numpy_dtype == dtypes.string and not isinstance(values, np.ndarray):
    proto_values = _FlattenToStrings(values)
    # At this point, values may be a list of objects that we could not
    # identify a common type for (hence it was inferred as
    # np.object_/dtypes.string).  If we are unable to convert it to a
    # string, we raise a more helpful error message.
    #
    # Ideally, we'd be able to convert the elements of the list to a
    # common type, but this type inference requires some thinking and
    # so we defer it for now.
    try:
      str_values = [compat.as_bytes(x) for x in proto_values]
    except TypeError:
      raise TypeError(f"Failed to convert elements of {values} to Tensor. "
                      "Consider casting elements to a supported type. See "
                      "https://www.tensorflow.org/api_docs/python/tf/dtypes "
                      "for supported TF dtypes.")
    tensor_proto.string_val.extend(str_values)
    return tensor_proto
  # TensorFlow expects C order (a.k.a., eigen row major).
  proto_values = nparray.ravel()
  append_fn = GetNumpyAppendFn(proto_values.dtype)
  if append_fn is None:
    raise TypeError(
        f"Element type not supported in TensorProto: {numpy_dtype.name}.")
  append_fn(tensor_proto, proto_values)
  return tensor_proto
