@tf_export("meshgrid")
@dispatch.add_dispatch_support
def meshgrid(*args, **kwargs):
  """Broadcasts parameters for evaluation on an N-D grid.
  Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
  of N-D coordinate arrays for evaluating expressions on an N-D grid.
  Notes:
  `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
  When the `indexing` argument is set to 'xy' (the default), the broadcasting
  instructions for the first two dimensions are swapped.
  Examples:
  Calling `X, Y = meshgrid(x, y)` with the tensors
  ```python
  x = [1, 2, 3]
  y = [4, 5, 6]
  X, Y = tf.meshgrid(x, y)
  # X = [[1, 2, 3],
  #      [1, 2, 3],
  #      [1, 2, 3]]
  # Y = [[4, 4, 4],
  #      [5, 5, 5],
  #      [6, 6, 6]]
  ```
  Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).
  Returns:
    outputs: A list of N `Tensor`s with rank N.
  Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
  """
  indexing = kwargs.pop("indexing", "xy")
  name = kwargs.pop("name", "meshgrid")
  if kwargs:
    key = list(kwargs.keys())[0]
    raise TypeError("'{}' is an invalid keyword argument "
                    "for this function".format(key))
  if indexing not in ("xy", "ij"):
    raise ValueError("Argument `indexing` parameter must be either "
                     f"'xy' or 'ij', got '{indexing}'")
  with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim
    if not ndim:
      return []
    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
      output.append(
          reshape(array_ops_stack.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [size(x) for x in args]
    output_dtype = ops.convert_to_tensor(args[0]).dtype.base_dtype
    if indexing == "xy" and ndim > 1:
      output[0] = reshape(output[0], (1, -1) + (1,) * (ndim - 2))
      output[1] = reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
      shapes[0], shapes[1] = shapes[1], shapes[0]
    # TODO(nolivia): improve performance with a broadcast
    mult_fact = ones(shapes, output_dtype)
    return [x * mult_fact for x in output]
