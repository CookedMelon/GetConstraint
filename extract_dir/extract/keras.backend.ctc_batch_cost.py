@keras_export("keras.backend.ctc_batch_cost")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.
    Args:
        y_true: tensor `(samples, max_string_length)`
            containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_true`.
    Returns:
        Tensor with shape (samples,1) containing the
            CTC loss of each element.
    """
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    sparse_labels = tf.cast(
        ctc_label_dense_to_sparse(y_true, label_length), tf.int32
    )
    y_pred = tf.math.log(
        tf.compat.v1.transpose(y_pred, perm=[1, 0, 2]) + epsilon()
    )
    return tf.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )
