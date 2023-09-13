@tf_export("data.experimental.from_variant")
def from_variant(variant, structure):
  """Constructs a dataset from the given variant and (nested) structure.
  Args:
    variant: A scalar `tf.variant` tensor representing a dataset.
    structure: A (nested) structure of `tf.TypeSpec` objects representing the
      structure of each element in the dataset.
  Returns:
    A `tf.data.Dataset` instance.
  """
  return _VariantDataset(variant, structure)  # pylint: disable=protected-access
