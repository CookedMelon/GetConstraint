@tf_export("edit_distance")
@dispatch.add_dispatch_support
def edit_distance(hypothesis, truth, normalize=True, name="edit_distance"):
  """Computes the Levenshtein distance between sequences.
  This operation takes variable-length sequences (`hypothesis` and `truth`),
  each provided as a `SparseTensor`, and computes the Levenshtein distance.
  You can normalize the edit distance by length of `truth` by setting
  `normalize` to true.
  For example:
  Given the following input,
  * `hypothesis` is a `tf.SparseTensor` of shape `[2, 1, 1]`
  * `truth` is a `tf.SparseTensor` of shape `[2, 2, 2]`
  >>> hypothesis = tf.SparseTensor(
  ...   [[0, 0, 0],
  ...    [1, 0, 0]],
  ...   ["a", "b"],
  ...   (2, 1, 1))
  >>> truth = tf.SparseTensor(
  ...   [[0, 1, 0],
  ...    [1, 0, 0],
  ...    [1, 0, 1],
  ...    [1, 1, 0]],
  ...    ["a", "b", "c", "a"],
  ...    (2, 2, 2))
  >>> tf.edit_distance(hypothesis, truth, normalize=True)
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[inf, 1. ],
         [0.5, 1. ]], dtype=float32)>
  The operation returns a dense Tensor of shape `[2, 2]` with
  edit distances normalized by `truth` lengths.
  **Note**: It is possible to calculate edit distance between two
  sparse tensors with variable-length values. However, attempting to create
  them while eager execution is enabled will result in a `ValueError`.
  For the following  inputs,
  ```python
  # 'hypothesis' is a tensor of shape `[2, 1]` with variable-length values:
  #   (0,0) = ["a"]
  #   (1,0) = ["b"]
  hypothesis = tf.sparse.SparseTensor(
      [[0, 0, 0],
       [1, 0, 0]],
      ["a", "b"],
      (2, 1, 1))
  # 'truth' is a tensor of shape `[2, 2]` with variable-length values:
  #   (0,0) = []
  #   (0,1) = ["a"]
  #   (1,0) = ["b", "c"]
  #   (1,1) = ["a"]
  truth = tf.sparse.SparseTensor(
      [[0, 1, 0],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1, 0]],
      ["a", "b", "c", "a"],
      (2, 2, 2))
  normalize = True
  # The output would be a dense Tensor of shape `(2,)`, with edit distances
  normalized by 'truth' lengths.
  # output => array([0., 0.5], dtype=float32)
  ```
  Args:
    hypothesis: A `SparseTensor` containing hypothesis sequences.
    truth: A `SparseTensor` containing truth sequences.
    normalize: A `bool`. If `True`, normalizes the Levenshtein distance by
      length of `truth.`
    name: A name for the operation (optional).
  Returns:
    A dense `Tensor` with rank `R - 1`, where R is the rank of the
    `SparseTensor` inputs `hypothesis` and `truth`.
  Raises:
    TypeError: If either `hypothesis` or `truth` are not a `SparseTensor`.
  """
  if not isinstance(
      hypothesis,
      (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
    raise TypeError("Hypothesis must be a SparseTensor.")
  if not isinstance(
      truth, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
    raise TypeError("Truth must be a SparseTensor.")
  return gen_array_ops.edit_distance(
      hypothesis.indices,
      hypothesis.values,
      hypothesis.dense_shape,
      truth.indices,
      truth.values,
      truth.dense_shape,
      normalize=normalize,
      name=name)
