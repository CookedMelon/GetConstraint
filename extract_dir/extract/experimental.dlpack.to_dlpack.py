@tf_export("experimental.dlpack.to_dlpack", v1=[])
def to_dlpack(tf_tensor):
  """Returns the dlpack capsule representing the tensor.
  This operation ensures the underlying data memory is ready when returns.
    ```python
    a = tf.tensor([1, 10])
    dlcapsule = tf.experimental.dlpack.to_dlpack(a)
    # dlcapsule represents the dlpack data structure
    ```
  Args:
    tf_tensor: Tensorflow eager tensor, to be converted to dlpack capsule.
  Returns:
    A PyCapsule named as dltensor, which shares the underlying memory to other
     framework. This PyCapsule can be consumed only once.
  """
  return pywrap_tfe.TFE_ToDlpackCapsule(tf_tensor)
