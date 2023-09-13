@tf_export("data.experimental.AutotuneAlgorithm")
class AutotuneAlgorithm(enum.Enum):
  """Represents the type of autotuning algorithm to use.
  DEFAULT: The default behavior is implementation specific and may change over
  time.
  HILL_CLIMB: In each optimization step, this algorithm chooses the optimial
  parameter and increases its value by 1.
  GRADIENT_DESCENT: In each optimization step, this algorithm updates the
  parameter values in the optimal direction.
  MAX_PARALLELISM: Similar to HILL_CLIMB but uses a relaxed stopping condition,
  allowing the optimization to oversubscribe the CPU.
  STAGE_BASED: In each optimization step, this algorithm chooses the worst
  bottleneck parameter and increases its value by 1.
  """
  DEFAULT = 0
  HILL_CLIMB = 1
  GRADIENT_DESCENT = 2
  MAX_PARALLELISM = 3
  STAGE_BASED = 4
  @classmethod
  def _to_proto(cls, obj):
    if obj == cls.DEFAULT:
      return model_pb2.AutotuneAlgorithm.DEFAULT
    if obj == cls.HILL_CLIMB:
      return model_pb2.AutotuneAlgorithm.HILL_CLIMB
    if obj == cls.GRADIENT_DESCENT:
      return model_pb2.AutotuneAlgorithm.GRADIENT_DESCENT
    if obj == cls.MAX_PARALLELISM:
      return model_pb2.AutotuneAlgorithm.MAX_PARALLELISM
    if obj == cls.STAGE_BASED:
      return model_pb2.AutotuneAlgorithm.STAGE_BASED
    raise ValueError(
        f"Invalid `obj.` Supported values include `DEFAULT`, `HILL_CLIMB` "
        f"`GRADIENT_DESCENT`, and `STAGE_BASED`. Got {obj.name}.")
  @classmethod
  def _from_proto(cls, pb):
    if pb == model_pb2.AutotuneAlgorithm.DEFAULT:
      return cls.DEFAULT
    if pb == model_pb2.AutotuneAlgorithm.HILL_CLIMB:
      return cls.HILL_CLIMB
    if pb == model_pb2.AutotuneAlgorithm.GRADIENT_DESCENT:
      return cls.GRADIENT_DESCENT
    if pb == model_pb2.AutotuneAlgorithm.MAX_PARALLELISM:
      return cls.MAX_PARALLELISM
    if pb == model_pb2.AutotuneAlgorithm.STAGE_BASED:
      return cls.STAGE_BASED
    raise ValueError(
        f"Invalid `pb.` Supported values include `DEFAULT`, `HILL_CLIMB`, "
        f"`GRADIENT_DESCENT` and `STAGE_BASED`. Got {pb}.")
