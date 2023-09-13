@tf_export("data.experimental.ExternalStatePolicy")
class ExternalStatePolicy(enum.Enum):
  """Represents how to handle external state during serialization.
  See the `tf.data.Options.experimental_external_state_policy` documentation
  for more information.
  """
  WARN = 0
  IGNORE = 1
  FAIL = 2
  @classmethod
  def _to_proto(cls, obj):
    """Convert enum to proto."""
    if obj == cls.IGNORE:
      return dataset_options_pb2.ExternalStatePolicy.POLICY_IGNORE
    if obj == cls.FAIL:
      return dataset_options_pb2.ExternalStatePolicy.POLICY_FAIL
    if obj == cls.WARN:
      return dataset_options_pb2.ExternalStatePolicy.POLICY_WARN
    raise ValueError(
        f"Invalid `obj.` Supported values include `POLICY_IGNORE`,"
        f"`POLICY_FAIL`, `POLICY_WARN`. Got {obj.name}.")
  @classmethod
  def _from_proto(cls, pb):
    """Convert proto to enum."""
    if pb == dataset_options_pb2.ExternalStatePolicy.POLICY_IGNORE:
      return cls.IGNORE
    if pb == dataset_options_pb2.ExternalStatePolicy.POLICY_FAIL:
      return cls.FAIL
    if pb == dataset_options_pb2.ExternalStatePolicy.POLICY_WARN:
      return cls.WARN
    raise ValueError(
        f"Invalid `pb.` Supported values include `POLICY_IGNORE`,"
        f"`POLICY_FAIL`, `POLICY_WARN`. Got {pb}.")
