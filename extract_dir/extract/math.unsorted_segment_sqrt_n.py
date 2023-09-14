@tf_export(
    "math.unsorted_segment_sqrt_n",
    v1=["math.unsorted_segment_sqrt_n", "unsorted_segment_sqrt_n"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("unsorted_segment_sqrt_n")
def unsorted_segment_sqrt_n(data, segment_ids, num_segments, name=None):
  r"""Computes the sum along segments of a tensor divided by the sqrt(N).
  Read [the section on
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.
  This operator is similar to the `tf.math.unsorted_segment_sum` operator.
  Additionally to computing the sum over segments, it divides the results by
  sqrt(N).
  \\(output_i = 1/sqrt(N_i) \sum_{j...} data[j...]\\) where the sum is over
  tuples `j...` such that `segment_ids[j...] == i` with \\N_i\\ being the
  number of occurrences of id \\i\\.
  If there is no entry for a given segment ID `i`, it outputs 0.
  Note that this op only supports floating point and complex dtypes,
  due to tf.sqrt only supporting these types.
  If the given segment ID `i` is negative, the value is dropped and will not
  be added to the sum of the segment.
  Caution: On CPU, values in `segment_ids` are always validated to be less than
  `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
  does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
  result in safe but unspecified behavior, which may include ignoring
  out-of-bound indices or outputting a tensor with a 0 stored in the first
  dimension of its shape if `num_segments` is 0.
  Args:
    data: A `Tensor` with floating point or complex dtype.
    segment_ids: An integer tensor whose shape is a prefix of `data.shape`.
      The values must be in the range `[0, num_segments)`.
      The values are always validated to be in range on CPU,
      never validated on GPU.
    num_segments: An integer scalar `Tensor`.  The number of distinct segment
      IDs.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`.  Has same shape as data, except for the first `segment_ids.rank`
    dimensions, which are replaced with a single dimension which has size
   `num_segments`.
  """
  with ops.name_scope(name, "UnsortedSegmentSqrtN"):
    data = ops.convert_to_tensor(data)
    segment_ids = ops.convert_to_tensor(segment_ids)
    N = _unsorted_segment_N(data, segment_ids, num_segments)
    summed = gen_math_ops.unsorted_segment_sum(data, segment_ids, num_segments)
    return summed / gen_math_ops.sqrt(N)
@tf_export(v1=["sparse.segment_sum", "sparse_segment_sum"])
@deprecation.deprecated_endpoints("sparse_segment_sum")
def sparse_segment_sum(data,
                       indices,
                       segment_ids,
                       name=None,
                       num_segments=None):
  r"""Computes the sum along sparse segments of a tensor.
  Read [the section on
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.
  Like `tf.math.segment_sum`, but `segment_ids` can have rank less than `data`'s
  first dimension, selecting a subset of dimension 0, specified by `indices`.
  `segment_ids` is allowed to have missing ids, in which case the output will
  be zeros at those indices. In those cases `num_segments` is used to determine
  the size of the output.
  For example:
  ```python
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
  # Select two rows, one segment.
  tf.sparse.segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
  # => [[0 0 0 0]]
  # Select two rows, two segment.
  tf.sparse.segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
  # => [[ 1  2  3  4]
  #     [-1 -2 -3 -4]]
  # With missing segment ids.
  tf.sparse.segment_sum(c, tf.constant([0, 1]), tf.constant([0, 2]),
                        num_segments=4)
  # => [[ 1  2  3  4]
  #     [ 0  0  0  0]
  #     [-1 -2 -3 -4]
  #     [ 0  0  0  0]]
  # Select all rows, two segments.
  tf.sparse.segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
  # => [[0 0 0 0]
  #     [5 6 7 8]]
  # Which is equivalent to:
  tf.math.segment_sum(c, tf.constant([0, 0, 1]))
  ```
  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`. Values
      should be sorted and can be repeated.
    name: A name for the operation (optional).
    num_segments: An optional int32 scalar. Indicates the size of the output
      `Tensor`.
  Returns:
    A `tensor` of the shape as data, except for dimension 0 which
    has size `k`, the number of segments specified via `num_segments` or
    inferred for the last element in `segments_ids`.
  """
  if num_segments is not None:
    return gen_math_ops.sparse_segment_sum_with_num_segments(
        data=data,
        indices=indices,
        segment_ids=segment_ids,
        num_segments=num_segments,
        name=name)
  else:
    return gen_math_ops.sparse_segment_sum(
        data=data, indices=indices, segment_ids=segment_ids, name=name)
