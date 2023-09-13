@tf_export("strings.reduce_join", v1=[])
@dispatch.add_dispatch_support
def reduce_join_v2(  # pylint: disable=missing-docstring
    inputs,
    axis=None,
    keepdims=False,
    separator="",
    name=None):
  """Joins all strings into a single string, or joins along an axis.
  This is the reduction operation for the elementwise `tf.strings.join` op.
  >>> tf.strings.reduce_join([['abc','123'],
  ...                         ['def','456']]).numpy()
  b'abc123def456'
  >>> tf.strings.reduce_join([['abc','123'],
  ...                         ['def','456']], axis=-1).numpy()
  array([b'abc123', b'def456'], dtype=object)
  >>> tf.strings.reduce_join([['abc','123'],
  ...                         ['def','456']],
  ...                        axis=-1,
  ...                        separator=" ").numpy()
  array([b'abc 123', b'def 456'], dtype=object)
  Args:
    inputs: A `tf.string` tensor.
    axis: Which axis to join along. The default behavior is to join all
      elements, producing a scalar.
    keepdims: If true, retains reduced dimensions with length 1.
    separator: a string added between each string being joined.
    name: A name for the operation (optional).
  Returns:
    A `tf.string` tensor.
  """
  with ops.name_scope(None, "ReduceJoin", [inputs, axis]):
    inputs_t = ops.convert_to_tensor(inputs)
    axis = _reduce_join_reduction_dims(inputs_t, axis)
    return gen_string_ops.reduce_join(
        inputs=inputs_t,
        reduction_indices=axis,
        keep_dims=keepdims,
        separator=separator,
        name=name)
