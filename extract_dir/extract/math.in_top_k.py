@tf_export("math.in_top_k", "nn.in_top_k", v1=[])
@dispatch.add_dispatch_support
def in_top_k_v2(targets, predictions, k, name=None):
  """Outputs whether the targets are in the top `K` predictions.
  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is finite (not inf, -inf, or nan) and among
  the top `k` predictions among all predictions for example `i`.
  `predictions` does not have to be normalized.
  Note that the behavior of `InTopK` differs from the `TopK` op in its handling
  of ties; if multiple classes have the same prediction value and straddle the
  top-`k` boundary, all of those classes are considered to be in the top `k`.
  >>> target = tf.constant([0, 1, 3])
  >>> pred = tf.constant([
  ...  [1.2, -0.3, 2.8, 5.2],
  ...  [0.1, 0.0, 0.0, 0.0],
  ...  [0.0, 0.5, 0.3, 0.3]],
  ...  dtype=tf.float32)
  >>> print(tf.math.in_top_k(target, pred, 2))
  tf.Tensor([False  True  True], shape=(3,), dtype=bool)
  Args:
    targets: A `batch_size` vector of class ids. Must be `int32` or `int64`.
    predictions: A `batch_size` x `classes` tensor of type `float32`.
    k: An `int`. The parameter to specify search space.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` with the same shape of `targets` with type of `bool`. Each
      element specifies if the target falls into top-k predictions.
  """
  return in_top_k(predictions, targets, k, name)
