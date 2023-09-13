@tf_export("TypeSpec", v1=["TypeSpec", "data.experimental.Structure"])
class TypeSpec(
    internal.TypeSpec,
    trace.TraceType,
    trace_type.Serializable,
    metaclass=abc.ABCMeta,
