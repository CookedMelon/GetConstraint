@tf_export("types.experimental.ConcreteFunction", v1=[])
class ConcreteFunction(Callable):
  """Base class for graph functions.
  A `ConcreteFunction` encapsulates a single graph function definition and
  is differentiable under `tf.GradientTape` contexts.
  """
