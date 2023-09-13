@tf_export("strings.split", v1=[])
@dispatch.add_dispatch_support
def string_split_v2(input, sep=None, maxsplit=-1, name=None):  # pylint: disable=redefined-builtin
  """Split elements of `input` based on `sep` into a `RaggedTensor`.
  Let N be the size of `input` (typically N will be the batch size). Split each
  element of `input` based on `sep` and return a `RaggedTensor` containing the
  split tokens. Empty tokens are ignored.
  Example:
  >>> tf.strings.split('hello world').numpy()
   array([b'hello', b'world'], dtype=object)
  >>> tf.strings.split(['hello world', 'a b c'])
  <tf.RaggedTensor [[b'hello', b'world'], [b'a', b'b', b'c']]>
  If `sep` is given, consecutive delimiters are not grouped together and are
  deemed to delimit empty strings. For example, `input` of `"1<>2<><>3"` and
  `sep` of `"<>"` returns `["1", "2", "", "3"]`. If `sep` is None or an empty
  string, consecutive whitespace are regarded as a single separator, and the
  result will contain no empty strings at the start or end if the string has
  leading or trailing whitespace.
  Note that the above mentioned behavior matches python's str.split.
  Args:
    input: A string `Tensor` of rank `N`, the strings to split.  If
      `rank(input)` is not known statically, then it is assumed to be `1`.
    sep: `0-D` string `Tensor`, the delimiter string.
    maxsplit: An `int`. If `maxsplit > 0`, limit of the split of the result.
    name: A name for the operation (optional).
  Raises:
    ValueError: If sep is not a string.
  Returns:
    A `RaggedTensor` of rank `N+1`, the strings split according to the
    delimiter.
  """
  with ops.name_scope(name, "StringSplit", [input]):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, dtype=dtypes.string, name="input")
    if isinstance(input, ragged_tensor.RaggedTensor):
      return input.with_flat_values(
          string_split_v2(input.flat_values, sep, maxsplit))
    rank = input.shape.ndims
    if rank == 0:
      return string_split_v2(array_ops_stack.stack([input]), sep, maxsplit)[0]
    elif rank == 1 or rank is None:
      sparse_result = string_ops.string_split_v2(
          input, sep=sep, maxsplit=maxsplit)
      return ragged_tensor.RaggedTensor.from_value_rowids(
          values=sparse_result.values,
          value_rowids=sparse_result.indices[:, 0],
          nrows=sparse_result.dense_shape[0],
          validate=False)
    else:
      return string_split_v2(
          ragged_tensor.RaggedTensor.from_tensor(input), sep, maxsplit)
