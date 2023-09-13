@tf_export("reverse_sequence", v1=[])
@dispatch.add_dispatch_support
def reverse_sequence_v2(input,
                        seq_lengths,
                        seq_axis=None,
                        batch_axis=None,
                        name=None):
  """Reverses variable length slices.
  This op first slices `input` along the dimension `batch_axis`, and for
  each slice `i`, reverses the first `seq_lengths[i]` elements along the
  dimension `seq_axis`.
  The elements of `seq_lengths` must obey `seq_lengths[i] <=
  input.dims[seq_axis]`, and `seq_lengths` must be a vector of length
  `input.dims[batch_axis]`.
  The output slice `i` along dimension `batch_axis` is then given by
  input slice `i`, with the first `seq_lengths[i]` slices along
  dimension `seq_axis` reversed.
  Example usage:
  >>> seq_lengths = [7, 2, 3, 5]
  >>> input = [[1, 2, 3, 4, 5, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0],
  ...          [1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]]
  >>> output = tf.reverse_sequence(input, seq_lengths, seq_axis=1, batch_axis=0)
  >>> output
  <tf.Tensor: shape=(4, 8), dtype=int32, numpy=
  array([[0, 0, 5, 4, 3, 2, 1, 0],
         [2, 1, 0, 0, 0, 0, 0, 0],
         [3, 2, 1, 4, 0, 0, 0, 0],
         [5, 4, 3, 2, 1, 6, 7, 8]], dtype=int32)>
  Args:
    input: A `Tensor`. The input to reverse.
    seq_lengths: A `Tensor`. Must be one of the following types: `int32`,
      `int64`. 1-D with length `input.dims(batch_axis)` and `max(seq_lengths) <=
      input.dims(seq_axis)`
    seq_axis: An `int`. The dimension which is partially reversed.
    batch_axis: An optional `int`. Defaults to `0`. The dimension along which
      reversal is performed.
    name: A name for the operation (optional).
  Returns:
    A Tensor. Has the same type as input.
  """
  return gen_array_ops.reverse_sequence(
      input=input,
      seq_lengths=seq_lengths,
      seq_dim=seq_axis,
      batch_dim=batch_axis,
      name=name)
