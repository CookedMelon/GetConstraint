@tf_export("data.experimental.SqlDataset", v1=[])
class SqlDatasetV2(dataset_ops.DatasetSource):
  """A `Dataset` consisting of the results from a SQL query.
  `SqlDataset` allows a user to read data from the result set of a SQL query.
  For example:
  ```python
  dataset = tf.data.experimental.SqlDataset("sqlite", "/foo/bar.sqlite3",
                                            "SELECT name, age FROM people",
                                            (tf.string, tf.int32))
  # Prints the rows of the result set of the above query.
  for element in dataset:
    print(element)
  ```
  """
  def __init__(self, driver_name, data_source_name, query, output_types):
    """Creates a `SqlDataset`.
    Args:
      driver_name: A 0-D `tf.string` tensor containing the database type.
        Currently, the only supported value is 'sqlite'.
      data_source_name: A 0-D `tf.string` tensor containing a connection string
        to connect to the database.
      query: A 0-D `tf.string` tensor containing the SQL query to execute.
      output_types: A tuple of `tf.DType` objects representing the types of the
        columns returned by `query`.
    """
    self._driver_name = ops.convert_to_tensor(
        driver_name, dtype=dtypes.string, name="driver_name")
    self._data_source_name = ops.convert_to_tensor(
        data_source_name, dtype=dtypes.string, name="data_source_name")
    self._query = ops.convert_to_tensor(
        query, dtype=dtypes.string, name="query")
    self._element_spec = nest.map_structure(
        lambda dtype: tensor_spec.TensorSpec([], dtype), output_types)
    variant_tensor = gen_experimental_dataset_ops.sql_dataset(
        self._driver_name, self._data_source_name, self._query,
        **self._flat_structure)
    super(SqlDatasetV2, self).__init__(variant_tensor)
  @property
  def element_spec(self):
    return self._element_spec
