@tf_export(
    "math.unsorted_segment_mean",
    v1=["math.unsorted_segment_mean", "unsorted_segment_mean"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("unsorted_segment_mean")
def unsorted_segment_mean(data, segment_ids, num_segments, name=None):
  r"""Computes the mean along segments of a tensor.
  Read [the section on
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.
  This operator is similar to the `tf.math.unsorted_segment_sum` operator.
  Instead of computing the sum over segments, it computes the mean of all
  entries belonging to a segment such that:
  \\(output_i = 1/N_i \sum_{j...} data[j...]\\) where the sum is over tuples
  `j...` such that `segment_ids[j...] == i` with \\N_i\\ being the number of
  occurrences of id \\i\\.
  If there is no entry for a given segment ID `i`, it outputs 0.
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
      The values must be less than `num_segments`.
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
  with ops.name_scope(name, "UnsortedSegmentMean"):
    data = ops.convert_to_tensor(data)
    segment_ids = ops.convert_to_tensor(segment_ids)
    N = _unsorted_segment_N(data, segment_ids, num_segments)
    summed = gen_math_ops.unsorted_segment_sum(data, segment_ids, num_segments)
    return summed / N
