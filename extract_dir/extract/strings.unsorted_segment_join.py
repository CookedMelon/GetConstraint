@tf_export("strings.unsorted_segment_join")
@dispatch.add_dispatch_support
def unsorted_segment_join(inputs,
                          segment_ids,
                          num_segments,
                          separator="",
                          name=None):
  """Joins the elements of `inputs` based on `segment_ids`.
  Computes the string join along segments of a tensor.
  Given `segment_ids` with rank `N` and `data` with rank `N+M`:
  ```
  output[i, k1...kM] = strings.join([data[j1...jN, k1...kM])
  ```
  where the join is over all `[j1...jN]` such that `segment_ids[j1...jN] = i`.
  Strings are joined in row-major order.
  For example:
  >>> inputs = ['this', 'a', 'test', 'is']
  >>> segment_ids = [0, 1, 1, 0]
  >>> num_segments = 2
  >>> separator = ' '
  >>> tf.strings.unsorted_segment_join(inputs, segment_ids, num_segments,
  ...                                  separator).numpy()
  array([b'this is', b'a test'], dtype=object)
  >>> inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
  >>> segment_ids = [1, 0, 1]
  >>> num_segments = 2
  >>> tf.strings.unsorted_segment_join(inputs, segment_ids, num_segments,
  ...                                  separator=':').numpy()
  array([[b'Y', b'6', b'6'],
         [b'Y:p', b'q:G', b'c:a']], dtype=object)
  Args:
    inputs: A list of `tf.Tensor` objects of type `tf.string`.
    segment_ids: A tensor whose shape is a prefix of `inputs.shape` and whose
      type must be `tf.int32` or `tf.int64`. Negative segment ids are not
      supported.
    num_segments: A scalar of type `tf.int32` or `tf.int64`. Must be
      non-negative and larger than any segment id.
    separator: The separator to use when joining. Defaults to `""`.
    name: A name for the operation (optional).
  Returns:
    A `tf.string` tensor representing the concatenated values, using the given
    separator.
  """
  return gen_string_ops.unsorted_segment_join(
      inputs, segment_ids, num_segments, separator=separator, name=name)
