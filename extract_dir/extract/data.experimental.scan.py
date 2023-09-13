@tf_export("data.experimental.scan")
def scan(initial_state, scan_func):
  """A transformation that scans a function across an input dataset.
  This transformation is a stateful relative of `tf.data.Dataset.map`.
  In addition to mapping `scan_func` across the elements of the input dataset,
  `scan()` accumulates one or more state tensors, whose initial values are
  `initial_state`.
  Args:
    initial_state: A nested structure of tensors, representing the initial state
      of the accumulator.
    scan_func: A function that maps `(old_state, input_element)` to
      `(new_state, output_element)`. It must take two arguments and return a
      pair of nested structures of tensors. The `new_state` must match the
      structure of `initial_state`.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.scan(initial_state=initial_state, scan_func=scan_func)
  return _apply_fn
