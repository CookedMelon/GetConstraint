@tf_export("constant", v1=[])
def constant(value, dtype=None, shape=None, name="Const"):
  """Creates a constant tensor from a tensor-like object.
  Note: All eager `tf.Tensor` values are immutable (in contrast to
  `tf.Variable`). There is nothing especially _constant_ about the value
  returned from `tf.constant`. This function is not fundamentally different from
  `tf.convert_to_tensor`. The name `tf.constant` comes from the `value` being
  embedded in a `Const` node in the `tf.Graph`. `tf.constant` is useful
  for asserting that the value can be embedded that way.
  If the argument `dtype` is not specified, then the type is inferred from
  the type of `value`.
  >>> # Constant 1-D Tensor from a python list.
  >>> tf.constant([1, 2, 3, 4, 5, 6])
  <tf.Tensor: shape=(6,), dtype=int32,
      numpy=array([1, 2, 3, 4, 5, 6], dtype=int32)>
  >>> # Or a numpy array
  >>> a = np.array([[1, 2, 3], [4, 5, 6]])
  >>> tf.constant(a)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
    array([[1, 2, 3],
           [4, 5, 6]])>
  If `dtype` is specified, the resulting tensor values are cast to the requested
  `dtype`.
  >>> tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float64)
  <tf.Tensor: shape=(6,), dtype=float64,
      numpy=array([1., 2., 3., 4., 5., 6.])>
  If `shape` is set, the `value` is reshaped to match. Scalars are expanded to
  fill the `shape`:
  >>> tf.constant(0, shape=(2, 3))
    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)>
  >>> tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)>
  `tf.constant` has no effect if an eager Tensor is passed as the `value`, it
  even transmits gradients:
  >>> v = tf.Variable([0.0])
  >>> with tf.GradientTape() as g:
  ...     loss = tf.constant(v + v)
  >>> g.gradient(loss, v).numpy()
  array([2.], dtype=float32)
  But, since `tf.constant` embeds the value in the `tf.Graph` this fails for
  symbolic tensors:
  >>> with tf.compat.v1.Graph().as_default():
  ...   i = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float32)
  ...   t = tf.constant(i)
  Traceback (most recent call last):
  ...
  TypeError: ...
  `tf.constant` will create tensors on the current device. Inputs which are
  already tensors maintain their placements unchanged.
  Related Ops:
  * `tf.convert_to_tensor` is similar but:
    * It has no `shape` argument.
    * Symbolic tensors are allowed to pass through.
    >>> with tf.compat.v1.Graph().as_default():
    ...   i = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float32)
    ...   t = tf.convert_to_tensor(i)
  * `tf.fill`: differs in a few ways:
    *   `tf.constant` supports arbitrary constants, not just uniform scalar
        Tensors like `tf.fill`.
    *   `tf.fill` creates an Op in the graph that is expanded at runtime, so it
        can efficiently represent large tensors.
    *   Since `tf.fill` does not embed the value, it can produce dynamically
        sized outputs.
  Args:
    value: A constant value (or list) of output type `dtype`.
    dtype: The type of the elements of the resulting tensor.
    shape: Optional dimensions of resulting tensor.
    name: Optional name for the tensor.
  Returns:
    A Constant Tensor.
  Raises:
    TypeError: if shape is incorrectly specified or unsupported.
    ValueError: if called on a symbolic tensor.
  """
  return _constant_impl(value, dtype, shape, name, verify_shape=False,
                        allow_broadcast=True)
