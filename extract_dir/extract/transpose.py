@tf_export("transpose", v1=[])
@dispatch.add_dispatch_support
def transpose_v2(a, perm=None, conjugate=False, name="transpose"):
  """Transposes `a`, where `a` is a Tensor.
  Permutes the dimensions according to the value of `perm`.
  The returned tensor's dimension `i` will correspond to the input dimension
  `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is the rank
  of the input tensor. Hence, by default, this operation performs a regular
  matrix transpose on 2-D input Tensors.
  If conjugate is `True` and `a.dtype` is either `complex64` or `complex128`
  then the values of `a` are conjugated and transposed.
  @compatibility(numpy)
  In `numpy` transposes are memory-efficient constant time operations as they
  simply return a new view of the same data with adjusted `strides`.
  TensorFlow does not support strides, so `transpose` returns a new tensor with
  the items permuted.
  @end_compatibility
  For example:
  >>> x = tf.constant([[1, 2, 3], [4, 5, 6]])
  >>> tf.transpose(x)
  <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
  array([[1, 4],
         [2, 5],
         [3, 6]], dtype=int32)>
  Equivalently, you could call `tf.transpose(x, perm=[1, 0])`.
  If `x` is complex, setting conjugate=True gives the conjugate transpose:
  >>> x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
  ...                  [4 + 4j, 5 + 5j, 6 + 6j]])
  >>> tf.transpose(x, conjugate=True)
  <tf.Tensor: shape=(3, 2), dtype=complex128, numpy=
  array([[1.-1.j, 4.-4.j],
         [2.-2.j, 5.-5.j],
         [3.-3.j, 6.-6.j]])>
  'perm' is more useful for n-dimensional tensors where n > 2:
  >>> x = tf.constant([[[ 1,  2,  3],
  ...                   [ 4,  5,  6]],
  ...                  [[ 7,  8,  9],
  ...                   [10, 11, 12]]])
  As above, simply calling `tf.transpose` will default to `perm=[2,1,0]`.
  To take the transpose of the matrices in dimension-0 (such as when you are
  transposing matrices where 0 is the batch dimension), you would set
  `perm=[0,2,1]`.
  >>> tf.transpose(x, perm=[0, 2, 1])
  <tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
  array([[[ 1,  4],
          [ 2,  5],
          [ 3,  6]],
          [[ 7, 10],
          [ 8, 11],
          [ 9, 12]]], dtype=int32)>
  Note: This has a shorthand `linalg.matrix_transpose`):
  Args:
    a: A `Tensor`.
    perm: A permutation of the dimensions of `a`.  This should be a vector.
    conjugate: Optional bool. Setting it to `True` is mathematically equivalent
      to tf.math.conj(tf.transpose(input)).
    name: A name for the operation (optional).
  Returns:
    A transposed `Tensor`.
  """
  return transpose(a=a, perm=perm, name=name, conjugate=conjugate)
