@tf_export(
    "tensor_scatter_nd_update",
    v1=["tensor_scatter_nd_update", "tensor_scatter_update"])
@dispatch.add_dispatch_support
def tensor_scatter_nd_update(tensor, indices, updates, name=None):
  """Scatter `updates` into an existing tensor according to `indices`.
  This operation creates a new tensor by applying sparse `updates` to the
  input `tensor`. This is similar to an index assignment.
  ```
  # Not implemented: tensors cannot be updated inplace.
  tensor[indices] = updates
  ```
  If an out of bound index is found on CPU, an error is returned.
  > **WARNING**: There are some GPU specific semantics for this operation.
  >
  > - If an out of bound index is found, the index is ignored.
  > - The order in which updates are applied is nondeterministic, so the output
  >   will be nondeterministic if `indices` contains duplicates.
  This operation is very similar to `tf.scatter_nd`, except that the updates are
  scattered onto an existing tensor (as opposed to a zero-tensor). If the memory
  for the existing tensor cannot be re-used, a copy is made and updated.
  In general:
  * `indices` is an integer tensor - the indices to update in `tensor`.
  * `indices` has **at least two** axes, the last axis is the depth of the
    index vectors.
  * For each index vector in `indices` there is a corresponding entry in
    `updates`.
  * If the length of the index vectors matches the rank of the `tensor`, then
    the index vectors each point to scalars in `tensor` and each update is a
    scalar.
  * If the length of the index vectors is less than the rank of `tensor`, then
    the index vectors each point to the slices of `tensor` and shape of the updates
    must match that slice.
  Overall this leads to the following shape constraints:
  ```
  assert tf.rank(indices) >= 2
  index_depth = indices.shape[-1]
  batch_shape = indices.shape[:-1]
  assert index_depth <= tf.rank(tensor)
  outer_shape = tensor.shape[:index_depth]
  inner_shape = tensor.shape[index_depth:]
  assert updates.shape == batch_shape + inner_shape
  ```
  Typical usage is often much simpler than this general form, and it
  can be better understood starting with simple examples:
  ### Scalar updates
  The simplest usage inserts scalar elements into a tensor by index.
  In this case, the `index_depth` must equal the rank of the
  input `tensor`, slice each column of `indices` is an index into an axis of the
  input `tensor`.
  In this simplest case the shape constraints are:
  ```
  num_updates, index_depth = indices.shape.as_list()
  assert updates.shape == [num_updates]
  assert index_depth == tf.rank(tensor)`
  ```
  For example, to insert 4 scattered elements in a rank-1 tensor with
  8 elements.
  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%"
    src="https://www.tensorflow.org/images/ScatterNd1.png">
  </div>
  This scatter operation would look like this:
  >>> tensor = [0, 0, 0, 0, 0, 0, 0, 0]    # tf.rank(tensor) == 1
  >>> indices = [[1], [3], [4], [7]]       # num_updates == 4, index_depth == 1
  >>> updates = [9, 10, 11, 12]            # num_updates == 4
  >>> print(tf.tensor_scatter_nd_update(tensor, indices, updates))
  tf.Tensor([ 0 9  0 10  11  0  0 12], shape=(8,), dtype=int32)
  The length (first axis) of `updates` must equal the length of the `indices`:
  `num_updates`. This is the number of updates being inserted. Each scalar
  update is inserted into `tensor` at the indexed location.
  For a higher rank input `tensor` scalar updates can be inserted by using an
  `index_depth` that matches `tf.rank(tensor)`:
  >>> tensor = [[1, 1], [1, 1], [1, 1]]    # tf.rank(tensor) == 2
  >>> indices = [[0, 1], [2, 0]]           # num_updates == 2, index_depth == 2
  >>> updates = [5, 10]                    # num_updates == 2
  >>> print(tf.tensor_scatter_nd_update(tensor, indices, updates))
  tf.Tensor(
      [[ 1  5]
       [ 1  1]
       [10  1]], shape=(3, 2), dtype=int32)
  ### Slice updates
  When the input `tensor` has more than one axis scatter can be used to update
  entire slices.
  In this case it's helpful to think of the input `tensor` as being a two level
  array-of-arrays. The shape of this two level array is split into the
  `outer_shape` and the `inner_shape`.
  `indices` indexes into the outer level of the input tensor (`outer_shape`).
  and replaces the sub-array at that location with the corresponding item from
  the `updates` list. The shape of each update is `inner_shape`.
  When updating a list of slices the shape constraints are:
  ```
  num_updates, index_depth = indices.shape.as_list()
  outer_shape = tensor.shape[:index_depth]
  inner_shape = tensor.shape[index_depth:]
  assert updates.shape == [num_updates, inner_shape]
  ```
  For example, to update rows of a `(6, 3)` `tensor`:
  >>> tensor = tf.zeros([6, 3], dtype=tf.int32)
  Use an index depth of one.
  >>> indices = tf.constant([[2], [4]])     # num_updates == 2, index_depth == 1
  >>> num_updates, index_depth = indices.shape.as_list()
  The `outer_shape` is `6`, the inner shape is `3`:
  >>> outer_shape = tensor.shape[:index_depth]
  >>> inner_shape = tensor.shape[index_depth:]
  2 rows are being indexed so 2 `updates` must be supplied.
  Each update must be shaped to match the `inner_shape`.
  >>> # num_updates == 2, inner_shape==3
  >>> updates = tf.constant([[1, 2, 3],
  ...                        [4, 5, 6]])
  Altogether this gives:
  >>> tf.tensor_scatter_nd_update(tensor, indices, updates).numpy()
  array([[0, 0, 0],
         [0, 0, 0],
         [1, 2, 3],
         [0, 0, 0],
         [4, 5, 6],
         [0, 0, 0]], dtype=int32)
  #### More slice update examples
  A tensor representing a batch of uniformly sized video clips naturally has 5
  axes: `[batch_size, time, width, height, channels]`.
  For example:
  >>> batch_size, time, width, height, channels = 13,11,7,5,3
  >>> video_batch = tf.zeros([batch_size, time, width, height, channels])
  To replace a selection of video clips:
    * Use an `index_depth` of 1 (indexing the `outer_shape`: `[batch_size]`)
    * Provide updates each with a shape matching the `inner_shape`:
      `[time, width, height, channels]`.
  To replace the first two clips with ones:
  >>> indices = [[0],[1]]
  >>> new_clips = tf.ones([2, time, width, height, channels])
  >>> tf.tensor_scatter_nd_update(video_batch, indices, new_clips)
  To replace a selection of frames in the videos:
  * `indices` must have an `index_depth` of 2 for the `outer_shape`:
    `[batch_size, time]`.
  * `updates` must be shaped like a list of images.  Each update must have a
    shape, matching the `inner_shape`: `[width, height, channels]`.
  To replace the first frame of the first three video clips:
  >>> indices = [[0, 0], [1, 0], [2, 0]] # num_updates=3, index_depth=2
  >>> new_images = tf.ones([
  ...   # num_updates=3, inner_shape=(width, height, channels)
  ...   3, width, height, channels])
  >>> tf.tensor_scatter_nd_update(video_batch, indices, new_images)
  ### Folded indices
  In simple cases it's convenient to think of `indices` and `updates` as
  lists, but this is not a strict requirement. Instead of a flat `num_updates`,
  the `indices` and `updates` can be folded into a `batch_shape`. This
  `batch_shape` is all axes of the `indices`, except for the innermost
  `index_depth` axis.
  ```
  index_depth = indices.shape[-1]
  batch_shape = indices.shape[:-1]
  ```
  Note: The one exception is that the `batch_shape` cannot be `[]`. You can't
  update a single index by passing indices with shape `[index_depth]`.
  `updates` must have a matching `batch_shape` (the axes before `inner_shape`).
  ```
  assert updates.shape == batch_shape + inner_shape
  ```
  Note: The result is equivalent to flattening the `batch_shape` axes of
  `indices` and `updates`. This generalization just avoids the need
  for reshapes when it is more natural to construct "folded" indices and
  updates.
  With this generalization the full shape constraints are:
  ```
  assert tf.rank(indices) >= 2
  index_depth = indices.shape[-1]
  batch_shape = indices.shape[:-1]
  assert index_depth <= tf.rank(tensor)
  outer_shape = tensor.shape[:index_depth]
  inner_shape = tensor.shape[index_depth:]
  assert updates.shape == batch_shape + inner_shape
  ```
  For example, to draw an `X` on a `(5,5)` matrix start with these indices:
  >>> tensor = tf.zeros([5,5])
  >>> indices = tf.constant([
  ...  [[0,0],
  ...   [1,1],
  ...   [2,2],
  ...   [3,3],
  ...   [4,4]],
  ...  [[0,4],
  ...   [1,3],
  ...   [2,2],
  ...   [3,1],
  ...   [4,0]],
  ... ])
  >>> indices.shape.as_list()  # batch_shape == [2, 5], index_depth == 2
  [2, 5, 2]
  Here the `indices` do not have a shape of `[num_updates, index_depth]`, but a
  shape of `batch_shape+[index_depth]`.
  Since the `index_depth` is equal to the rank of `tensor`:
  * `outer_shape` is `(5,5)`
  * `inner_shape` is `()` - each update is scalar
  * `updates.shape` is `batch_shape + inner_shape == (5,2) + ()`
  >>> updates = [
  ...   [1,1,1,1,1],
  ...   [1,1,1,1,1],
  ... ]
  Putting this together gives:
  >>> tf.tensor_scatter_nd_update(tensor, indices, updates).numpy()
  array([[1., 0., 0., 0., 1.],
         [0., 1., 0., 1., 0.],
         [0., 0., 1., 0., 0.],
         [0., 1., 0., 1., 0.],
         [1., 0., 0., 0., 1.]], dtype=float32)
  Args:
    tensor: Tensor to copy/update.
    indices: Indices to update.
    updates: Updates to apply at the indices.
    name: Optional name for the operation.
  Returns:
    A new tensor with the given shape and updates applied according to the
    indices.
  """
  return gen_array_ops.tensor_scatter_update(
      tensor=tensor, indices=indices, updates=updates, name=name)
# Define quantize_v2 here in order to make name the second-to-last attribute,
# because round_mode was added later.
# (And also now because of 'axis' processing).
@tf_export(v1=["quantize_v2"])
@dispatch.add_dispatch_support
@deprecation.deprecated(
    "2017-10-25",
    "`tf.quantize_v2` is deprecated, please use `tf.quantization.quantize` "
    "instead.")  # pylint: disable=missing-docstring
def quantize_v2(
    input,  # pylint: disable=redefined-builtin
    min_range,
    max_range,
    T,
    mode="MIN_COMBINED",
    name=None,
    round_mode="HALF_AWAY_FROM_ZERO",
    narrow_range=False,
    axis=None,
    ensure_minimum_range=0.01):
  if axis is None:
    axis = -1
  elif axis < 0:
    if input.shape.ndims is None:
      raise ValueError("input should have known rank to use negative axis.")
    axis %= input.shape.ndims
  if ensure_minimum_range != 0.01:
    return gen_array_ops.quantize_v2(
        input,
        min_range,
        max_range,
        T=T,
        mode=mode,
        name=name,
        round_mode=round_mode,
        narrow_range=narrow_range,
        axis=axis,
        ensure_minimum_range=ensure_minimum_range)
  return gen_array_ops.quantize_v2(
      input,
      min_range,
      max_range,
      T=T,
      mode=mode,
      name=name,
      round_mode=round_mode,
      narrow_range=narrow_range,
      axis=axis)
quantize_v2.__doc__ = """Please use `tf.quantization.quantize` instead."""
# We want to expose tf.quantization.quantize instead of
# tf.quantization.quantize; we can deprecate tf.quantization.quantize in next
# version of TensorFlow.
