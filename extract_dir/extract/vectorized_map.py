@tf_export("vectorized_map")
def vectorized_map(fn, elems, fallback_to_while_loop=True, warn=True):
  """Parallel map on the list of tensors unpacked from `elems` on dimension 0.
  This method works similar to `tf.map_fn` but is optimized to run much faster,
  possibly with a much larger memory footprint. The speedups are obtained by
  vectorization (see [Auto-Vectorizing TensorFlow Graphs: Jacobians,
  Auto-Batching and Beyond](https://arxiv.org/pdf/1903.04243.pdf)). The idea
  behind vectorization is to semantically launch all the invocations of `fn` in
  parallel and fuse corresponding operations across all these invocations. This
  fusion is done statically at graph generation time and the generated code is
  often similar in performance to a manually fused version.
  Because `tf.vectorized_map` fully parallelizes the batch, this method will
  generally be significantly faster than using `tf.map_fn`, especially in eager
  mode. However this is an experimental feature and currently has a lot of
  limitations:
    - There should be no data dependency between the different semantic
      invocations of `fn`, i.e. it should be safe to map the elements of the
      inputs in any order.
    - Stateful kernels may mostly not be supported since these often imply a
      data dependency. We do support a limited set of such stateful kernels
      though (like RandomFoo, Variable operations like reads, etc).
    - `fn` has limited support for control flow operations.
    - `fn` should return nested structure of Tensors or Operations. However
      if an Operation is returned, it should have zero outputs.
    - The shape and dtype of any intermediate or output tensors in the
      computation of `fn` should not depend on the input to `fn`.
  Examples:
  ```python
  def outer_product(a):
    return tf.tensordot(a, a, 0)
  batch_size = 100
  a = tf.ones((batch_size, 32, 32))
  c = tf.vectorized_map(outer_product, a)
  assert c.shape == (batch_size, 32, 32, 32, 32)
  ```
  ```python
  # Computing per-example gradients
  batch_size = 10
  num_features = 32
  layer = tf.keras.layers.Dense(1)
  def model_fn(arg):
    with tf.GradientTape() as g:
      inp, label = arg
      inp = tf.expand_dims(inp, 0)
      label = tf.expand_dims(label, 0)
      prediction = layer(inp)
      loss = tf.nn.l2_loss(label - prediction)
    return g.gradient(loss, (layer.kernel, layer.bias))
  inputs = tf.random.uniform([batch_size, num_features])
  labels = tf.random.uniform([batch_size, 1])
  per_example_gradients = tf.vectorized_map(model_fn, (inputs, labels))
  assert per_example_gradients[0].shape == (batch_size, num_features, 1)
  assert per_example_gradients[1].shape == (batch_size, 1)
  ```
  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same (possibly nested) structure as `elems`, and returns a possibly
      nested structure of Tensors and Operations, which may be different than
      the structure of `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unpacked along their first dimension. The nested sequence of the
      resulting slices will be mapped over by `fn`. The first dimensions of all
      elements must broadcast to a consistent value; equivalently, each
      element tensor must have first dimension of either `B` or `1`, for some
      common batch size `B >= 1`.
    fallback_to_while_loop: If true, on failing to vectorize an operation,
      the unsupported op is wrapped in a tf.while_loop to execute the map
      iterations. Note that this fallback only happens for unsupported ops and
      other parts of `fn` are still vectorized. If false, on encountering an
      unsupported op, a ValueError is thrown. Note that the fallbacks can result
      in slowdowns since vectorization often yields speedup of one to two orders
      of magnitude.
    warn: If set to `false`, this will supress any warnings due to operation
    conversions in the provided `fn` falling back to while loops.
  Returns:
    A tensor or (possibly nested) sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.
    Although they are less common as user-visible inputs and outputs, note that
    tensors of type `tf.variant` which represent tensor lists (for example from
    `tf.raw_ops.TensorListFromTensor`) are vectorized by stacking the list
    contents rather than the variant itself, and so the container tensor will
    have a scalar shape when returned rather than the usual stacked shape. This
    improves the performance of control flow gradient vectorization.
  Raises:
    ValueError: If vectorization fails and fallback_to_while_loop is False.
  """
  elems = variable_utils.convert_variables_to_tensors(elems)
  elems = nest.map_structure(ops.convert_to_tensor,
                             elems,
                             expand_composites=True)
  def loop_fn(i):
    gathered_elems = nest.map_structure(
        lambda x: _gather_from_tensor_or_composite(x, i), elems)
    return fn(gathered_elems)
  # Extract batch size from the maximum first dimension of any element.
  flat_elems = nest.flatten(
      nest.map_structure(
          functools.partial(_composite_to_tensors,
                            is_batched=True),
          elems))
  def _get_shape(x):
    if x.shape.rank is None:
      return None
    return x.shape.as_list()[0]
  static_first_dims = [_get_shape(elem) for elem in flat_elems]
  if any(s is None for s in static_first_dims):
    batch_size = math_ops.reduce_max(
        [array_ops.shape(elem)[0] for elem in flat_elems])
  else:
    batch_size = max(static_first_dims)
  return pfor(
      loop_fn,
      batch_size,
      fallback_to_while_loop=fallback_to_while_loop,
      warn=warn)
