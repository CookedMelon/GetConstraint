@keras_export("keras.backend.ctc_label_dense_to_sparse")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def ctc_label_dense_to_sparse(labels, label_lengths):
    """Converts CTC labels from dense to sparse.
    Args:
        labels: dense CTC labels.
        label_lengths: length of the labels.
    Returns:
        A sparse tensor representation of the labels.
    """
    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])
    def range_less_than(old_input, current_input):
        return tf.expand_dims(tf.range(tf.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )
    init = tf.cast(tf.fill([1, label_shape[1]], 0), tf.bool)
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]
    label_array = tf.reshape(
        tf.tile(tf.range(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)
    batch_array = tf.compat.v1.transpose(
        tf.reshape(
            tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns),
            reverse(label_shape, 0),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = tf.compat.v1.transpose(
        tf.reshape(concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )
    vals_sparse = tf.compat.v1.gather_nd(labels, indices)
    return tf.SparseTensor(
        tf.cast(indices, tf.int64), vals_sparse, tf.cast(label_shape, tf.int64)
    )
