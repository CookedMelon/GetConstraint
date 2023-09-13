@tf_export("io.VarLenFeature", v1=["VarLenFeature", "io.VarLenFeature"])
class VarLenFeature(collections.namedtuple("VarLenFeature", ["dtype"])):
  """Configuration for parsing a variable-length input feature.
  Fields:
    dtype: Data type of input.
  """
  pass
