@tf_export("nn.scale_regularization_loss")
@dispatch.add_dispatch_support
def scale_regularization_loss(regularization_loss):
  """Scales the sum of the given regularization losses by number of replicas.
  Usage with distribution strategy and custom training loop:
  ```python
  with strategy.scope():
    def compute_loss(self, label, predictions):
      per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)
      # Compute loss that is scaled by sample_weight and by global batch size.
      loss = tf.nn.compute_average_loss(
          per_example_loss,
          sample_weight=sample_weight,
          global_batch_size=GLOBAL_BATCH_SIZE)
      # Add scaled regularization losses.
      loss += tf.nn.scale_regularization_loss(tf.nn.l2_loss(weights))
      return loss
  ```
  Args:
    regularization_loss: Regularization loss.
  Returns:
    Scalar loss value.
  """  # pylint: disable=g-doc-exception
  if (distribute_lib.has_strategy()
      and distribute_lib.in_cross_replica_context()):
    raise RuntimeError(
        "You are calling `scale_regularization_loss` in cross replica context, "
        "while it was expected to be called in replica context.")
  num_replicas = distribute_lib.get_strategy().num_replicas_in_sync
  return math_ops.reduce_sum(regularization_loss) / num_replicas
