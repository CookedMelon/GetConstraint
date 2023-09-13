"/home/cc/Workspace/tfconstraint/python/ops/array_ops.py"
@tf_export(
    "linalg.tensor_diag_part", v1=["linalg.tensor_diag_part", "diag_part"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("diag_part")
def tensor_diag_part(
    input,  # pylint:disable=redefined-builtin
    name=None):
  """Returns the diagonal part of the tensor.
  This operation returns a tensor with the `diagonal` part
  of the `input`. The `diagonal` part is computed as follows:
  Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
  tensor of rank `k` with dimensions `[D1,..., Dk]` where:
  `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
  For a rank 2 tensor, `linalg.diag_part` and `linalg.tensor_diag_part`
  produce the same result. For rank 3 and higher, linalg.diag_part extracts
  the diagonal of each inner-most matrix in the tensor. An example where
  they differ is given below.
  >>> x = [[[[1111,1112],[1121,1122]],
  ...       [[1211,1212],[1221,1222]]],
  ...      [[[2111, 2112], [2121, 2122]],
  ...       [[2211, 2212], [2221, 2222]]]
  ...      ]
  >>> tf.linalg.tensor_diag_part(x)
  <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
  array([[1111, 1212],
         [2121, 2222]], dtype=int32)>
  >>> tf.linalg.diag_part(x).shape
  TensorShape([2, 2, 2])
  Args:
    input: A `Tensor` with rank `2k`.
    name: A name for the operation (optional).
  Returns:
    A Tensor containing diagonals of `input`. Has the same type as `input`, and
    rank `k`.
  """
  return gen_array_ops.diag_part(input=input, name=name)
