@tf_export("math.approx_min_k", "nn.approx_min_k")
@dispatch.add_dispatch_support
def approx_min_k(operand,
                 k,
                 reduction_dimension=-1,
                 recall_target=0.95,
                 reduction_input_size_override=-1,
                 aggregate_to_topk=True,
                 name=None):
  """Returns min `k` values and their indices of the input `operand` in an approximate manner.
  See https://arxiv.org/abs/2206.14286 for the algorithm details. This op is
  only optimized on TPU currently.
  Args:
    operand : Array to search for min-k. Must be a floating number type.
    k : Specifies the number of min-k.
    reduction_dimension: Integer dimension along which to search. Default: -1.
    recall_target: Recall target for the approximation.
    reduction_input_size_override : When set to a positive value, it overrides
      the size determined by `operand[reduction_dim]` for evaluating the recall.
      This option is useful when the given `operand` is only a subset of the
      overall computation in SPMD or distributed pipelines, where the true input
      size cannot be deferred by the `operand` shape.
    aggregate_to_topk: When true, aggregates approximate results to top-k. When
      false, returns the approximate results. The number of the approximate
      results is implementation defined and is greater equals to the specified
      `k`.
    name: Optional name for the operation.
  Returns:
    Tuple of two arrays. The arrays are the least `k` values and the
    corresponding indices along the `reduction_dimension` of the input
    `operand`.  The arrays' dimensions are the same as the input `operand`
    except for the `reduction_dimension`: when `aggregate_to_topk` is true,
    the reduction dimension is `k`; otherwise, it is greater equals to `k`
    where the size is implementation-defined.
  We encourage users to wrap `approx_min_k` with jit. See the following example
  for nearest neighbor search over the squared l2 distance:
  >>> import tensorflow as tf
  >>> @tf.function(jit_compile=True)
  ... def l2_ann(qy, db, half_db_norms, k=10, recall_target=0.95):
  ...   dists = half_db_norms - tf.einsum('ik,jk->ij', qy, db)
  ...   return tf.nn.approx_min_k(dists, k=k, recall_target=recall_target)
  >>>
  >>> qy = tf.random.uniform((256,128))
  >>> db = tf.random.uniform((2048,128))
  >>> half_db_norms = tf.norm(db, axis=1) / 2
  >>> dists, neighbors = l2_ann(qy, db, half_db_norms)
  In the example above, we compute `db_norms/2 - dot(qy, db^T)` instead of
  `qy^2 - 2 dot(qy, db^T) + db^2` for performance reason. The former uses less
  arithmetics and produces the same set of neighbors.
  """
  return gen_nn_ops.approx_top_k(
      operand,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=False,
      reduction_input_size_override=reduction_input_size_override,
      aggregate_to_topk=aggregate_to_topk,
      name=name)
