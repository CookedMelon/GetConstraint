@tf_export("types.experimental.TracingContext", v1=[])
class TracingContext(metaclass=abc.ABCMeta):
  """Contains information scoped to the tracing of multiple objects.
  `TracingContext` is a container class for flags and variables that have
  any kind of influence on the tracing behaviour of the class implementing
  the __tf_tracing_type__. This context will be shared across all
  __tf_tracing_type__ calls while constructing the TraceType for a particular
  set of objects.
  """
