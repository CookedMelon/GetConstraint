@tf_export("io.FixedLenFeature", v1=["io.FixedLenFeature", "FixedLenFeature"])
class FixedLenFeature(collections.namedtuple(
    "FixedLenFeature", ["shape", "dtype", "default_value"])):
  """Configuration for parsing a fixed-length input feature.
  To treat sparse input as dense, provide a `default_value`; otherwise,
  the parse functions will fail on any examples missing this feature.
  Fields:
    shape: Shape of input data.
    dtype: Data type of input.
    default_value: Value to be used if an example is missing this feature. It
        must be compatible with `dtype` and of the specified `shape`.
  """
  def __new__(cls, shape, dtype, default_value=None):
    return super(FixedLenFeature, cls).__new__(
        cls, shape, dtype, default_value)
