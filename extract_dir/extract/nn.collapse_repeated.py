@tf_export("nn.collapse_repeated")
@dispatch.add_dispatch_support
def collapse_repeated(labels, seq_length, name=None):
  """Merge repeated labels into single labels.
  Args:
    labels: Tensor of shape [batch, max value in seq_length]
    seq_length: Tensor of shape [batch], sequence length of each batch element.
    name: A name for this `Op`. Defaults to "collapse_repeated_labels".
  Returns:
    A tuple `(collapsed_labels, new_seq_length)` where
    collapsed_labels: Tensor of shape [batch, max_seq_length] with repeated
    labels collapsed and padded to max_seq_length, eg:
    `[[A, A, B, B, A], [A, B, C, D, E]] => [[A, B, A, 0, 0], [A, B, C, D, E]]`
    new_seq_length: int tensor of shape [batch] with new sequence lengths.
  """
  with ops.name_scope(name, "collapse_repeated_labels", [labels, seq_length]):
    labels = ops.convert_to_tensor(labels, name="labels")
    seq_length = ops.convert_to_tensor(seq_length, name="seq_length")
    # Mask labels that don't equal previous label.
    label_mask = array_ops.concat([
        array_ops.ones_like(labels[:, :1], dtypes.bool),
        math_ops.not_equal(labels[:, 1:], labels[:, :-1])
    ],
                                  axis=1)
    # Filter labels that aren't in the original sequence.
    maxlen = _get_dim(labels, 1)
    seq_mask = array_ops.sequence_mask(seq_length, maxlen=maxlen)
    label_mask = math_ops.logical_and(label_mask, seq_mask)
    # Count masks for new sequence lengths.
    new_seq_len = math_ops.reduce_sum(
        math_ops.cast(label_mask, dtypes.int32), axis=1)
    # Mask indexes based on sequence length mask.
    new_maxlen = math_ops.reduce_max(new_seq_len)
    idx_mask = array_ops.sequence_mask(new_seq_len, maxlen=new_maxlen)
    # Flatten everything and mask out labels to keep and sparse indices.
    flat_labels = array_ops.reshape(labels, [-1])
    flat_label_mask = array_ops.reshape(label_mask, [-1])
    flat_idx_mask = array_ops.reshape(idx_mask, [-1])
    idx = math_ops.range(_get_dim(flat_idx_mask, 0))
    # Scatter to flat shape.
    flat = array_ops.scatter_nd(
        indices=array_ops.expand_dims(
            array_ops.boolean_mask(idx, flat_idx_mask), axis=1),
        updates=array_ops.boolean_mask(flat_labels, flat_label_mask),
        shape=array_ops.shape(flat_idx_mask))
    # Reshape back to square batch.
    batch_size = _get_dim(labels, 0)
    new_shape = [batch_size, new_maxlen]
    return (array_ops.reshape(flat, new_shape),
            math_ops.cast(new_seq_len, seq_length.dtype))
