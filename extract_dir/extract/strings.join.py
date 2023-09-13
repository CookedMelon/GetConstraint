@tf_export("strings.join", v1=["strings.join", "string_join"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("string_join")
def string_join(inputs, separator="", name=None):
  """Perform element-wise concatenation of a list of string tensors.
  Given a list of string tensors of same shape, performs element-wise
  concatenation of the strings of the same index in all tensors.
  >>> tf.strings.join(['abc','def']).numpy()
  b'abcdef'
  >>> tf.strings.join([['abc','123'],
  ...                  ['def','456'],
  ...                  ['ghi','789']]).numpy()
  array([b'abcdefghi', b'123456789'], dtype=object)
  >>> tf.strings.join([['abc','123'],
  ...                  ['def','456']],
  ...                  separator=" ").numpy()
  array([b'abc def', b'123 456'], dtype=object)
  The reduction version of this elementwise operation is
  `tf.strings.reduce_join`
  Args:
    inputs: A list of `tf.Tensor` objects of same size and `tf.string` dtype.
    separator: A string added between each string being joined.
    name: A name for the operation (optional).
  Returns:
    A `tf.string` tensor.
  """
  return gen_string_ops.string_join(inputs, separator=separator, name=name)
