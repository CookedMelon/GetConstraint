@tf_export("data.experimental.to_variant")
def to_variant(dataset):
  """Returns a variant representing the given dataset.
  Args:
    dataset: A `tf.data.Dataset`.
  Returns:
    A scalar `tf.variant` tensor representing the given dataset.
  """
  return dataset._variant_tensor  # pylint: disable=protected-access
