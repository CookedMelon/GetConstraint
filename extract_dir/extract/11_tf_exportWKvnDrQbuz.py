"/home/cc/Workspace/tfconstraint/python/ops/check_ops.py"
@tf_export(
    'debugging.assert_same_float_dtype',
    v1=['debugging.assert_same_float_dtype', 'assert_same_float_dtype'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_same_float_dtype')
def assert_same_float_dtype(tensors=None, dtype=None):
  """Validate and return float type based on `tensors` and `dtype`.
  For ops such as matrix multiplication, inputs and weights must be of the
  same float type. This function validates that all `tensors` are the same type,
  validates that type is `dtype` (if supplied), and returns the type. Type must
  be a floating point type. If neither `tensors` nor `dtype` is supplied,
  the function will return `dtypes.float32`.
  Args:
    tensors: Tensors of input values. Can include `None` elements, which will be
        ignored.
    dtype: Expected type.
  Returns:
    Validated type.
  Raises:
    ValueError: if neither `tensors` nor `dtype` is supplied, or result is not
        float, or the common type of the inputs is not a floating point type.
  """
  if tensors:
    dtype = _assert_same_base_type(tensors, dtype)
  if not dtype:
    dtype = dtypes.float32
  elif not dtype.is_floating:
    raise ValueError('Expected floating point type, got %s.' % dtype)
  return dtype
@tf_export('debugging.assert_scalar', v1=[])
@dispatch.add_dispatch_support
def assert_scalar_v2(tensor, message=None, name=None):
  """Asserts that the given `tensor` is a scalar.
  This function raises `ValueError` unless it can be certain that the given
  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is
  unknown.
  This is always checked statically, so this method returns nothing.
  Args:
    tensor: A `Tensor`.
    message: A string to prefix to the default message.
    name:  A name for this operation. Defaults to "assert_scalar"
  Raises:
    ValueError: If the tensor is not scalar (rank 0), or if its shape is
      unknown.
  """
  assert_scalar(tensor=tensor, message=message, name=name)
@tf_export(v1=['debugging.assert_scalar', 'assert_scalar'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_scalar')
def assert_scalar(tensor, name=None, message=None):
  """Asserts that the given `tensor` is a scalar (i.e. zero-dimensional).
  This function raises `ValueError` unless it can be certain that the given
  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is
  unknown.
  Args:
    tensor: A `Tensor`.
    name:  A name for this operation. Defaults to "assert_scalar"
    message: A string to prefix to the default message.
  Returns:
    The input tensor (potentially converted to a `Tensor`).
  Raises:
    ValueError: If the tensor is not scalar (rank 0), or if its shape is
      unknown.
  """
  with ops.name_scope(name, 'assert_scalar', [tensor]) as name_scope:
    tensor = ops.convert_to_tensor(tensor, name=name_scope)
    shape = tensor.get_shape()
    message = _message_prefix(message)
    if shape.ndims != 0:
      if context.executing_eagerly():
        raise ValueError('%sExpected scalar shape, saw shape: %s.'
                         % (message, shape,))
      else:
        raise ValueError('%sExpected scalar shape for %s, saw shape: %s.'
                         % (message, tensor.name, shape))
    return tensor
def _message_prefix(message):
  if message:
    return '%s.  ' % message
  return ''
@tf_export('ensure_shape')
@dispatch.add_dispatch_support
def ensure_shape(x, shape, name=None):
  """Updates the shape of a tensor and checks at runtime that the shape holds.
  When executed, this operation asserts that the input tensor `x`'s shape
  is compatible with the `shape` argument.
  See `tf.TensorShape.is_compatible_with` for details.
  >>> x = tf.constant([[1, 2, 3],
  ...                  [4, 5, 6]])
  >>> x = tf.ensure_shape(x, [2, 3])
  Use `None` for unknown dimensions:
  >>> x = tf.ensure_shape(x, [None, 3])
  >>> x = tf.ensure_shape(x, [2, None])
  If the tensor's shape is not compatible with the `shape` argument, an error
  is raised:
  >>> x = tf.ensure_shape(x, [5])
  Traceback (most recent call last):
  ...
  tf.errors.InvalidArgumentError: Shape of tensor dummy_input [3] is not
    compatible with expected shape [5]. [Op:EnsureShape]
  During graph construction (typically tracing a `tf.function`),
  `tf.ensure_shape` updates the static-shape of the **result** tensor by
  merging the two shapes. See `tf.TensorShape.merge_with` for details.
  This is most useful when **you** know a shape that can't be determined
  statically by TensorFlow.
  The following trivial `tf.function` prints the input tensor's
  static-shape before and after `ensure_shape` is applied.
  >>> @tf.function
  ... def f(tensor):
  ...   print("Static-shape before:", tensor.shape)
  ...   tensor = tf.ensure_shape(tensor, [None, 3])
  ...   print("Static-shape after:", tensor.shape)
  ...   return tensor
  This lets you see the effect of `tf.ensure_shape` when the function is traced:
  >>> cf = f.get_concrete_function(tf.TensorSpec([None, None]))
  Static-shape before: (None, None)
  Static-shape after: (None, 3)
  >>> cf(tf.zeros([3, 3])) # Passes
  >>> cf(tf.constant([1, 2, 3])) # fails
  Traceback (most recent call last):
  ...
  InvalidArgumentError:  Shape of tensor x [3] is not compatible with expected shape [3,3].
  The above example raises `tf.errors.InvalidArgumentError`, because `x`'s
  shape, `(3,)`, is not compatible with the `shape` argument, `(None, 3)`
  Inside a `tf.function` or `v1.Graph` context it checks both the buildtime and
  runtime shapes. This is stricter than `tf.Tensor.set_shape` which only
  checks the buildtime shape.
  Note: This differs from `tf.Tensor.set_shape` in that it sets the static shape
  of the resulting tensor and enforces it at runtime, raising an error if the
  tensor's runtime shape is incompatible with the specified shape.
  `tf.Tensor.set_shape` sets the static shape of the tensor without enforcing it
  at runtime, which may result in inconsistencies between the statically-known
  shape of tensors and the runtime value of tensors.
  For example, of loading images of a known size:
  >>> @tf.function
  ... def decode_image(png):
  ...   image = tf.image.decode_png(png, channels=3)
  ...   # the `print` executes during tracing.
  ...   print("Initial shape: ", image.shape)
  ...   image = tf.ensure_shape(image,[28, 28, 3])
  ...   print("Final shape: ", image.shape)
  ...   return image
  When tracing a function, no ops are being executed, shapes may be unknown.
  See the [Concrete Functions Guide](https://www.tensorflow.org/guide/concrete_function)
  for details.
  >>> concrete_decode = decode_image.get_concrete_function(
  ...     tf.TensorSpec([], dtype=tf.string))
  Initial shape:  (None, None, 3)
  Final shape:  (28, 28, 3)
  >>> image = tf.random.uniform(maxval=255, shape=[28, 28, 3], dtype=tf.int32)
  >>> image = tf.cast(image,tf.uint8)
  >>> png = tf.image.encode_png(image)
  >>> image2 = concrete_decode(png)
  >>> print(image2.shape)
  (28, 28, 3)
  >>> image = tf.concat([image,image], axis=0)
  >>> print(image.shape)
  (56, 28, 3)
  >>> png = tf.image.encode_png(image)
  >>> image2 = concrete_decode(png)
  Traceback (most recent call last):
  ...
  tf.errors.InvalidArgumentError:  Shape of tensor DecodePng [56,28,3] is not
    compatible with expected shape [28,28,3].
  Caution: if you don't use the result of `tf.ensure_shape` the check may not
  run.
  >>> @tf.function
  ... def bad_decode_image(png):
  ...   image = tf.image.decode_png(png, channels=3)
  ...   # the `print` executes during tracing.
  ...   print("Initial shape: ", image.shape)
  ...   # BAD: forgot to use the returned tensor.
  ...   tf.ensure_shape(image,[28, 28, 3])
  ...   print("Final shape: ", image.shape)
  ...   return image
  >>> image = bad_decode_image(png)
  Initial shape:  (None, None, 3)
  Final shape:  (None, None, 3)
  >>> print(image.shape)
  (56, 28, 3)
  Args:
    x: A `Tensor`.
    shape: A `TensorShape` representing the shape of this tensor, a
      `TensorShapeProto`, a list, a tuple, or None.
    name: A name for this operation (optional). Defaults to "EnsureShape".
  Returns:
    A `Tensor`. Has the same type and contents as `x`.
  Raises:
    tf.errors.InvalidArgumentError: If `shape` is incompatible with the shape
    of `x`.
  """
  if not isinstance(shape, tensor_shape.TensorShape):
    shape = tensor_shape.TensorShape(shape)
  return array_ops.ensure_shape(x, shape, name=name)
@ops.RegisterGradient('EnsureShape')
def _ensure_shape_grad(op, grad):
  del op  # Unused.
  return grad
