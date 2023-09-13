@tf_export("nn.bias_add")
@dispatch.add_dispatch_support
def bias_add(value, bias, data_format=None, name=None):
  """Adds `bias` to `value`.
  This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.
  Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
  case where both types are quantized.
  Args:
    value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
      `int16`, `int8`, `complex64`, or `complex128`.
    bias: A 1-D `Tensor` with size matching the channel dimension of `value`.
      Must be the same type as `value` unless `value` is a quantized type,
      in which case a different quantized type may be used.
    data_format: A string. 'N...C' and 'NC...' are supported. If `None` (the
      default) is specified then 'N..C' is assumed.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` with the same type as `value`.
  Raises:
    ValueError if data format is unrecognized, if `value` has less than two
    dimensions when `data_format` is 'N..C'/`None` or `value` has less
    then three dimensions when `data_format` is `NC..`, if `bias` does not
    have exactly one dimension (is a vector), or if the size of `bias`
    does not match the size of the channel dimension of `value`.
  """
  with ops.name_scope(name, "BiasAdd", [value, bias]) as name:
    if data_format is not None:
      if data_format.startswith("NC"):
        data_format = "NCHW"
      elif data_format.startswith("N") and data_format.endswith("C"):
        data_format = "NHWC"
      else:
        raise ValueError("`data_format` must be of the form `N...C` or "
                         f"`NC...`. Received: data_format={data_format}")
    if not context.executing_eagerly():
      value = ops.convert_to_tensor(value, name="input")
      bias = ops.convert_to_tensor(bias, dtype=value.dtype, name="bias")
    return gen_nn_ops.bias_add(value, bias, data_format=data_format, name=name)
