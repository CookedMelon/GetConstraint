@tf_export("tuple", v1=[])
@dispatch.add_dispatch_support
def tuple_v2(tensors, control_inputs=None, name=None):
  """Groups tensors together.
  The returned tensors have the same value as the input tensors, but they
  are computed only after all the input tensors have been computed.
  Note: *In TensorFlow 2 with eager and/or Autograph, you should not require
  this method, as ops execute in the expected order thanks to automatic control
  dependencies.* Only use `tf.tuple` when working with v1 `tf.Graph` code.
  See also `tf.group` and `tf.control_dependencies`.
  Example:
  >>> with tf.Graph().as_default():
  ...   with tf.compat.v1.Session() as sess:
  ...     v = tf.Variable(0.0)
  ...     a = tf.constant(1.0)
  ...     sess.run(tf.compat.v1.global_variables_initializer())
  ...     for i in range(5):
  ...       update_op = v.assign_add(1.0)
  ...       b = a + v
  ...       res_b = sess.run(b)
  ...       res_v = sess.run(v)
  ...       print(res_v)
  0.0
  0.0
  0.0
  0.0
  0.0
  >>> with tf.Graph().as_default():
  ...   with tf.compat.v1.Session() as sess:
  ...     v = tf.Variable(0.0)
  ...     a = tf.constant(1.0)
  ...     sess.run(tf.compat.v1.global_variables_initializer())
  ...     for i in range(5):
  ...       update_op = v.assign_add(1.0)
  ...       calc = [a + v]
  ...       # `tf.tuple` ensures `update_op` is run before `b`
  ...       b = tf.tuple(calc, [tf.group(update_op)])
  ...       res_b = sess.run(b)
  ...       res_v = sess.run(v)
  ...       print(res_v)
  1.0
  2.0
  3.0
  4.0
  5.0
  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    control_inputs: List of additional ops to finish before returning.
    name: (optional) A name to use as a `name_scope` for the operation.
  Returns:
    Same as `tensors`.
  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
    TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
      objects.
  """
  return tuple(tensors=tensors, name=name, control_inputs=control_inputs)  # pylint: disable=redefined-builtin
