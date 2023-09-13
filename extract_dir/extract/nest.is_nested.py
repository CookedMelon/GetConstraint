@tf_export("nest.is_nested")
def is_nested(seq):
  """Returns true if its input is a nested structure.
  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a nested structure.
  Args:
    seq: the value to test.
  Returns:
    True if the input is a nested structure.
  """
  return nest_util.is_nested(nest_util.Modality.CORE, seq)
