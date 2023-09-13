@tf_export("Tensor", "experimental.numpy.ndarray", v1=["Tensor"])
class Tensor(
    pywrap_tf_session.PyTensor, internal.NativeObject, core_tf_types.Symbol
