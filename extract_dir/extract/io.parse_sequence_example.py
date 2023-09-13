@tf_export("io.parse_sequence_example")
@dispatch.add_dispatch_support
def parse_sequence_example(serialized,
                           context_features=None,
                           sequence_features=None,
                           example_names=None,
                           name=None):
  # pylint: disable=line-too-long
  """Parses a batch of `SequenceExample` protos.
  Parses a vector of serialized
  [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  protos given in `serialized`.
  This op parses serialized sequence examples into a tuple of dictionaries,
  each mapping keys to `Tensor` and `SparseTensor` objects.
  The first dictionary contains mappings for keys appearing in
  `context_features`, and the second dictionary contains mappings for keys
  appearing in `sequence_features`.
  At least one of `context_features` and `sequence_features` must be provided
  and non-empty.
  The `context_features` keys are associated with a `SequenceExample` as a
  whole, independent of time / frame.  In contrast, the `sequence_features` keys
  provide a way to access variable-length data within the `FeatureList` section
  of the `SequenceExample` proto.  While the shapes of `context_features` values
  are fixed with respect to frame, the frame dimension (the first dimension)
  of `sequence_features` values may vary between `SequenceExample` protos,
  and even between `feature_list` keys within the same `SequenceExample`.
  `context_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenFeature`  objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is  mapped to a `RaggedTensor`; and each
  `FixedLenFeature` is mapped to a `Tensor`, of the specified type, shape, and
  default value.
  `sequence_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenSequenceFeature` objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor`; and
  each `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified
  type. The shape will be `(B,T,) + df.dense_shape` for
  `FixedLenSequenceFeature` `df`, where `B` is the batch size, and `T` is the
  length of the associated `FeatureList` in the `SequenceExample`. For instance,
  `FixedLenSequenceFeature([])` yields a scalar 2-D `Tensor` of static shape
  `[None, None]` and dynamic shape `[B, T]`, while
  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 3-D matrix `Tensor`
  of static shape `[None, None, k]` and dynamic shape `[B, T, k]`.
  Like the input, the resulting output tensors have a batch dimension. This
  means that the original per-example shapes of `VarLenFeature`s and
  `FixedLenSequenceFeature`s can be lost. To handle that situation, this op also
  provides dicts of shape tensors as part of the output. There is one dict for
  the context features, and one for the feature_list features. Context features
  of type `FixedLenFeature`s will not be present, since their shapes are already
  known by the caller. In situations where the input `FixedLenSequenceFeature`s
  are of different sequence lengths across examples, the shorter examples will
  be padded with default datatype values: 0 for numeric types, and the empty
  string for string types.
  Each `SparseTensor` corresponding to `sequence_features` represents a ragged
  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`
  entry and `index` is the value's index in the list of values associated with
  that time.
  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`
  entries with `allow_missing=True` are optional; otherwise, we will fail if
  that `Feature` or `FeatureList` is missing from any example in `serialized`.
  `example_name` may contain a descriptive name for the corresponding serialized
  proto. This may be useful for debugging purposes, but it has no effect on the
  output. If not `None`, `example_name` must be a scalar.
  Args:
    serialized: A vector (1-D Tensor) of type string containing binary
      serialized `SequenceExample` protos.
    context_features: A mapping of feature keys to `FixedLenFeature` or
      `VarLenFeature` or `RaggedFeature` values. These features are associated
      with a `SequenceExample` as a whole.
    sequence_features: A mapping of feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` or `RaggedFeature` values.
      These features are associated with data within the `FeatureList` section
      of the `SequenceExample` proto.
    example_names: A vector (1-D Tensor) of strings (optional), the name of the
      serialized protos.
    name: A name for this operation (optional).
  Returns:
    A tuple of three `dict`s, each mapping keys to `Tensor`s,
    `SparseTensor`s, and `RaggedTensor`. The first dict contains the context
    key/values, the second dict contains the feature_list key/values, and the
    final dict contains the lengths of any dense feature_list features.
  Raises:
    ValueError: if any feature is invalid.
  """
  if not (context_features or sequence_features):
    raise ValueError("Both `context_features` and `sequence_features` argument "
                     "are None, but at least one should have values.")
  context_params = _ParseOpParams.from_features(
      context_features, [VarLenFeature, FixedLenFeature, RaggedFeature])
  feature_list_params = _ParseOpParams.from_features(
      sequence_features,
      [VarLenFeature, FixedLenSequenceFeature, RaggedFeature])
  with ops.name_scope(name, "ParseSequenceExample",
                      [serialized, example_names]):
    outputs = _parse_sequence_example_raw(serialized, example_names,
                                          context_params, feature_list_params,
                                          name)
    context_output, feature_list_output, feature_list_lengths = outputs
    if context_params.ragged_keys:
      context_output = _construct_tensors_for_composite_features(
          context_features, context_output)
    if feature_list_params.ragged_keys:
      feature_list_output = _construct_tensors_for_composite_features(
          sequence_features, feature_list_output)
    return context_output, feature_list_output, feature_list_lengths
