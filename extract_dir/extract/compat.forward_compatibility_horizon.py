@tf_export("compat.forward_compatibility_horizon")
@tf_contextlib.contextmanager
def forward_compatibility_horizon(year, month, day):
  """Context manager for testing forward compatibility of generated graphs.
  See [Version
  compatibility](https://www.tensorflow.org/guide/versions#backward_and_partial_forward_compatibility).
  To ensure forward compatibility of generated graphs (see `forward_compatible`)
  with older binaries, new features can be gated with:
  ```python
  if compat.forward_compatible(year=2018, month=08, date=01):
    generate_graph_with_new_features()
  else:
    generate_graph_so_older_binaries_can_consume_it()
  ```
  However, when adding new features, one may want to unittest it before
  the forward compatibility window expires. This context manager enables
  such tests. For example:
  ```python
  from tensorflow.python.compat import compat
  def testMyNewFeature(self):
    with compat.forward_compatibility_horizon(2018, 08, 02):
       # Test that generate_graph_with_new_features() has an effect
  ```
  Args:
    year:  A year (e.g., 2018). Must be an `int`.
    month: A month (1 <= month <= 12) in year. Must be an `int`.
    day:   A day (1 <= day <= 31, or 30, or 29, or 28) in month. Must be an
      `int`.
  Yields:
    Nothing.
  """
  try:
    _update_forward_compatibility_date_number(datetime.date(year, month, day))
    yield
  finally:
    _update_forward_compatibility_date_number()
