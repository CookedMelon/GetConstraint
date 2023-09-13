"/home/cc/Workspace/tfconstraint/python/feature_column/feature_column_v2.py"
@tf_export(
    'feature_column.make_parse_example_spec',
    v1=[])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def make_parse_example_spec_v2(feature_columns):
  """Creates parsing spec dictionary from input feature_columns.
  The returned dictionary can be used as arg 'features' in
  `tf.io.parse_example`.
  Typical usage example:
  ```python
  # Define features and transformations
  feature_a = tf.feature_column.categorical_column_with_vocabulary_file(...)
  feature_b = tf.feature_column.numeric_column(...)
  feature_c_bucketized = tf.feature_column.bucketized_column(
      tf.feature_column.numeric_column("feature_c"), ...)
  feature_a_x_feature_c = tf.feature_column.crossed_column(
      columns=["feature_a", feature_c_bucketized], ...)
  feature_columns = set(
      [feature_b, feature_c_bucketized, feature_a_x_feature_c])
  features = tf.io.parse_example(
      serialized=serialized_examples,
      features=tf.feature_column.make_parse_example_spec(feature_columns))
  ```
  For the above example, make_parse_example_spec would return the dict:
  ```python
  {
      "feature_a": parsing_ops.VarLenFeature(tf.string),
      "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
      "feature_c": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
  }
  ```
  Args:
    feature_columns: An iterable containing all feature columns. All items
      should be instances of classes derived from `FeatureColumn`.
  Returns:
    A dict mapping each feature key to a `FixedLenFeature` or `VarLenFeature`
    value.
  Raises:
    ValueError: If any of the given `feature_columns` is not a `FeatureColumn`
      instance.
  """
  result = {}
  for column in feature_columns:
    if not isinstance(column, FeatureColumn):
      raise ValueError('All feature_columns must be FeatureColumn instances. '
                       'Given: {}'.format(column))
    config = column.parse_example_spec
    for key, value in six.iteritems(config):
      if key in result and value != result[key]:
        raise ValueError('feature_columns contain different parse_spec for key '
                         '{}. Given {} and {}'.format(key, value, result[key]))
    result.update(config)
  return result
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.embedding_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def embedding_column(categorical_column,
                     dimension,
                     combiner='mean',
                     initializer=None,
                     ckpt_to_load_from=None,
                     tensor_name_in_ckpt=None,
                     max_norm=None,
                     trainable=True,
                     use_safe_embedding_lookup=True):
  """`DenseColumn` that converts from sparse, categorical input.
  Use this when your inputs are sparse, but you want to convert them to a dense
  representation (e.g., to feed to a DNN).
  Inputs must be a `CategoricalColumn` created by any of the
  `categorical_column_*` function. Here is an example of using
  `embedding_column` with `DNNClassifier`:
  ```python
  video_id = categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [embedding_column(video_id, 9),...]
  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)
  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels
  estimator.train(input_fn=input_fn, steps=100)
  ```
  Here is an example using `embedding_column` with model_fn:
  ```python
  def model_fn(features, ...):
    video_id = categorical_column_with_identity(
        key='video_id', num_buckets=1000000, default_value=0)
    columns = [embedding_column(video_id, 9),...]
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```
  Args:
    categorical_column: A `CategoricalColumn` created by a
      `categorical_column_with_*` function. This column produces the sparse IDs
      that are inputs to the embedding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, embedding values are l2-normalized to this value.
    trainable: Whether or not the embedding is trainable. Default is True.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.
  Returns:
    `DenseColumn` that converts from sparse input.
  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: If eager execution is enabled.
  """
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')
  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. '
                     'Embedding of column_name: {}'.format(
                         categorical_column.name))
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))
  return EmbeddingColumn(
      categorical_column=categorical_column,
      dimension=dimension,
      combiner=combiner,
      initializer=initializer,
      ckpt_to_load_from=ckpt_to_load_from,
      tensor_name_in_ckpt=tensor_name_in_ckpt,
      max_norm=max_norm,
      trainable=trainable,
      use_safe_embedding_lookup=use_safe_embedding_lookup)
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(v1=['feature_column.shared_embedding_columns'])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def shared_embedding_columns(categorical_columns,
                             dimension,
                             combiner='mean',
                             initializer=None,
                             shared_embedding_collection_name=None,
                             ckpt_to_load_from=None,
                             tensor_name_in_ckpt=None,
                             max_norm=None,
                             trainable=True,
                             use_safe_embedding_lookup=True):
  """List of dense columns that convert from sparse, categorical input.
  This is similar to `embedding_column`, except that it produces a list of
  embedding columns that share the same embedding weights.
  Use this when your inputs are sparse and of the same type (e.g. watched and
  impression video IDs that share the same vocabulary), and you want to convert
  them to a dense representation (e.g., to feed to a DNN).
  Inputs must be a list of categorical columns created by any of the
  `categorical_column_*` function. They must all be of the same type and have
  the same arguments except `key`. E.g. they can be
  categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
  all columns could also be weighted_categorical_column.
  Here is an example embedding of two features for a DNNClassifier model:
  ```python
  watched_video_id = categorical_column_with_vocabulary_file(
      'watched_video_id', video_vocabulary_file, video_vocabulary_size)
  impression_video_id = categorical_column_with_vocabulary_file(
      'impression_video_id', video_vocabulary_file, video_vocabulary_size)
  columns = shared_embedding_columns(
      [watched_video_id, impression_video_id], dimension=10)
  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)
  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels
  estimator.train(input_fn=input_fn, steps=100)
  ```
  Here is an example using `shared_embedding_columns` with model_fn:
  ```python
  def model_fn(features, ...):
    watched_video_id = categorical_column_with_vocabulary_file(
        'watched_video_id', video_vocabulary_file, video_vocabulary_size)
    impression_video_id = categorical_column_with_vocabulary_file(
        'impression_video_id', video_vocabulary_file, video_vocabulary_size)
    columns = shared_embedding_columns(
        [watched_video_id, impression_video_id], dimension=10)
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```
  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.
  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.
  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('shared_embedding_columns are not supported when eager '
                       'execution is enabled.')
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')
  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1. / math.sqrt(dimension))
  # Sort the columns so the default collection name is deterministic even if the
  # user passes columns from an unsorted collection, such as dict.values().
  sorted_columns = sorted(categorical_columns, key=lambda x: x.name)
  c0 = sorted_columns[0]
  num_buckets = c0._num_buckets  # pylint: disable=protected-access
  if not isinstance(c0, fc_old._CategoricalColumn):  # pylint: disable=protected-access
    raise ValueError(
        'All categorical_columns must be subclasses of _CategoricalColumn. '
        'Given: {}, of type: {}'.format(c0, type(c0)))
  while isinstance(
      c0,
      (
          fc_old._WeightedCategoricalColumn,  # pylint: disable=protected-access
          WeightedCategoricalColumn,
          fc_old._SequenceCategoricalColumn,  # pylint: disable=protected-access
          SequenceCategoricalColumn)):
    c0 = c0.categorical_column
  for c in sorted_columns[1:]:
    while isinstance(
        c,
        (
            fc_old._WeightedCategoricalColumn,  # pylint: disable=protected-access
            WeightedCategoricalColumn,
            fc_old._SequenceCategoricalColumn,  # pylint: disable=protected-access
            SequenceCategoricalColumn)):
      c = c.categorical_column
    if not isinstance(c, type(c0)):
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same type, or be weighted_categorical_column or sequence column '
          'of the same type. Given column: {} of type: {} does not match given '
          'column: {} of type: {}'.format(c0, type(c0), c, type(c)))
    if num_buckets != c._num_buckets:  # pylint: disable=protected-access
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same number of buckets. ven column: {} with buckets: {} does  '
          'not match column: {} with buckets: {}'.format(
              c0, num_buckets, c, c._num_buckets))  # pylint: disable=protected-access
  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'
  result = []
  for column in categorical_columns:
    result.append(
        fc_old._SharedEmbeddingColumn(  # pylint: disable=protected-access
            categorical_column=column,
            initializer=initializer,
            dimension=dimension,
            combiner=combiner,
            shared_embedding_collection_name=shared_embedding_collection_name,
            ckpt_to_load_from=ckpt_to_load_from,
            tensor_name_in_ckpt=tensor_name_in_ckpt,
            max_norm=max_norm,
            trainable=trainable,
            use_safe_embedding_lookup=use_safe_embedding_lookup))
  return result
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
