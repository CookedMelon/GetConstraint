"/home/cc/Workspace/tfconstraint/python/feature_column/feature_column_v2.py"
@tf_export(
    'feature_column.shared_embeddings',
    v1=[])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def shared_embedding_columns_v2(categorical_columns,
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
    shared_embedding_collection_name: Optional collective name of these columns.
      If not given, a reasonable name will be chosen based on the names of
      `categorical_columns`.
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
  num_buckets = c0.num_buckets
  if not isinstance(c0, CategoricalColumn):
    raise ValueError(
        'All categorical_columns must be subclasses of CategoricalColumn. '
        'Given: {}, of type: {}'.format(c0, type(c0)))
  while isinstance(c0, (WeightedCategoricalColumn, SequenceCategoricalColumn)):
    c0 = c0.categorical_column
  for c in sorted_columns[1:]:
    while isinstance(c, (WeightedCategoricalColumn, SequenceCategoricalColumn)):
      c = c.categorical_column
    if not isinstance(c, type(c0)):
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same type, or be weighted_categorical_column or sequence column '
          'of the same type. Given column: {} of type: {} does not match given '
          'column: {} of type: {}'.format(c0, type(c0), c, type(c)))
    if num_buckets != c.num_buckets:
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same number of buckets. Given column: {} with buckets: {} does  '
          'not match column: {} with buckets: {}'.format(
              c0, num_buckets, c, c.num_buckets))
  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'
  column_creator = SharedEmbeddingColumnCreator(
      dimension, initializer, ckpt_to_load_from, tensor_name_in_ckpt,
      num_buckets, trainable, shared_embedding_collection_name,
      use_safe_embedding_lookup)
  result = []
  for column in categorical_columns:
    result.append(
        column_creator(
            categorical_column=column, combiner=combiner, max_norm=max_norm))
  return result
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.numeric_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def numeric_column(key,
                   shape=(1,),
                   default_value=None,
                   dtype=dtypes.float32,
                   normalizer_fn=None):
  """Represents real valued or numerical features.
  Example:
  Assume we have data with two features `a` and `b`.
  >>> data = {'a': [15, 9, 17, 19, 21, 18, 25, 30],
  ...    'b': [5.0, 6.4, 10.5, 13.6, 15.7, 19.9, 20.3 , 0.0]}
  Let us represent the features `a` and `b` as numerical features.
  >>> a = tf.feature_column.numeric_column('a')
  >>> b = tf.feature_column.numeric_column('b')
  Feature column describe a set of transformations to the inputs.
  For example, to "bucketize" feature `a`, wrap the `a` column in a
  `feature_column.bucketized_column`.
  Providing `5` bucket boundaries, the bucketized_column api
  will bucket this feature in total of `6` buckets.
  >>> a_buckets = tf.feature_column.bucketized_column(a,
  ...    boundaries=[10, 15, 20, 25, 30])
  Create a `DenseFeatures` layer which will apply the transformations
  described by the set of `tf.feature_column` objects:
  >>> feature_layer = tf.keras.layers.DenseFeatures([a_buckets, b])
  >>> print(feature_layer(data))
  tf.Tensor(
  [[ 0.   0.   1.   0.   0.   0.   5. ]
   [ 1.   0.   0.   0.   0.   0.   6.4]
   [ 0.   0.   1.   0.   0.   0.  10.5]
   [ 0.   0.   1.   0.   0.   0.  13.6]
   [ 0.   0.   0.   1.   0.   0.  15.7]
   [ 0.   0.   1.   0.   0.   0.  19.9]
   [ 0.   0.   0.   0.   1.   0.  20.3]
   [ 0.   0.   0.   0.   0.   1.   0. ]], shape=(8, 7), dtype=float32)
  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    shape: An iterable of integers specifies the shape of the `Tensor`. An
      integer can be given which means a single dimension `Tensor` with given
      width. The `Tensor` representing the column will have the shape of
      [batch_size] + `shape`.
    default_value: A single value compatible with `dtype` or an iterable of
      values compatible with `dtype` which the column takes on during
      `tf.Example` parsing if data is missing. A default value of `None` will
      cause `tf.io.parse_example` to fail if an example does not contain this
      column. If a single value is provided, the same value will be applied as
      the default value for every item. If an iterable of values is provided,
      the shape of the `default_value` should be equal to the given `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
    normalizer_fn: If not `None`, a function that can be used to normalize the
      value of the tensor after `default_value` is applied for parsing.
      Normalizer function takes the input `Tensor` as its argument, and returns
      the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
      even though the most common use case of this function is normalization, it
      can be used for any kind of Tensorflow transformations.
  Returns:
    A `NumericColumn`.
  Raises:
    TypeError: if any dimension in shape is not an int
    ValueError: if any dimension in shape is not a positive integer
    TypeError: if `default_value` is an iterable but not compatible with `shape`
    TypeError: if `default_value` is not compatible with `dtype`.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  shape = _check_shape(shape, key)
  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError('dtype must be convertible to float. '
                     'dtype: {}, key: {}'.format(dtype, key))
  default_value = fc_utils.check_default_value(shape, default_value, dtype, key)
  if normalizer_fn is not None and not callable(normalizer_fn):
    raise TypeError(
        'normalizer_fn must be a callable. Given: {}'.format(normalizer_fn))
  fc_utils.assert_key_is_string(key)
  return NumericColumn(
      key,
      shape=shape,
      default_value=default_value,
      dtype=dtype,
      normalizer_fn=normalizer_fn)
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.bucketized_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def bucketized_column(source_column, boundaries):
  """Represents discretized dense input bucketed by `boundaries`.
  Buckets include the left boundary, and exclude the right boundary. Namely,
  `boundaries=[0., 1., 2.]` generates buckets `(-inf, 0.)`, `[0., 1.)`,
  `[1., 2.)`, and `[2., +inf)`.
  For example, if the inputs are
  ```python
  boundaries = [0, 10, 100]
  input tensor = [[-5, 10000]
                  [150,   10]
                  [5,    100]]
  ```
  then the output will be
  ```python
  output = [[0, 3]
            [3, 2]
            [1, 3]]
  ```
  Example:
  ```python
  price = tf.feature_column.numeric_column('price')
  bucketized_price = tf.feature_column.bucketized_column(
      price, boundaries=[...])
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(
      ..., features=tf.feature_column.make_parse_example_spec(columns))
  dense_tensor = tf.keras.layers.DenseFeatures(columns)(features)
  ```
  A `bucketized_column` can also be crossed with another categorical column
  using `crossed_column`:
  ```python
  price = tf.feature_column.numeric_column('price')
  # bucketized_column converts numerical feature to a categorical one.
  bucketized_price = tf.feature_column.bucketized_column(
      price, boundaries=[...])
  # 'keywords' is a string feature.
  price_x_keywords = tf.feature_column.crossed_column(
      [bucketized_price, 'keywords'], 50K)
  columns = [price_x_keywords, ...]
  features = tf.io.parse_example(
      ..., features=tf.feature_column.make_parse_example_spec(columns))
  dense_tensor = tf.keras.layers.DenseFeatures(columns)(features)
  linear_model = tf.keras.experimental.LinearModel(units=...)(dense_tensor)
  ```
  Args:
    source_column: A one-dimensional dense column which is generated with
      `numeric_column`.
    boundaries: A sorted list or tuple of floats specifying the boundaries.
  Returns:
    A `BucketizedColumn`.
  Raises:
    ValueError: If `source_column` is not a numeric column, or if it is not
      one-dimensional.
    ValueError: If `boundaries` is not a sorted list or tuple.
  """
  if not isinstance(source_column, (NumericColumn, fc_old._NumericColumn)):  # pylint: disable=protected-access
    raise ValueError(
        'source_column must be a column generated with numeric_column(). '
        'Given: {}'.format(source_column))
  if len(source_column.shape) > 1:
    raise ValueError('source_column must be one-dimensional column. '
                     'Given: {}'.format(source_column))
  if not boundaries:
    raise ValueError('boundaries must not be empty.')
  if not (isinstance(boundaries, list) or isinstance(boundaries, tuple)):
    raise ValueError('boundaries must be a sorted list.')
  for i in range(len(boundaries) - 1):
    if boundaries[i] >= boundaries[i + 1]:
      raise ValueError('boundaries must be a sorted list.')
  return BucketizedColumn(source_column, tuple(boundaries))
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.categorical_column_with_hash_bucket')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def categorical_column_with_hash_bucket(key,
                                        hash_bucket_size,
                                        dtype=dtypes.string):
  """Represents sparse feature where ids are set by hashing.
  Use this when your sparse features are in string or integer format, and you
  want to distribute your inputs into a finite number of buckets by hashing.
  output_id = Hash(input_feature_string) % bucket_size for string type input.
  For int type input, the value is converted to its string representation first
  and then hashed by the same formula.
  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.
  Example:
  ```python
  import tensorflow as tf
  keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
  10000)
  columns = [keywords]
  features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
  'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
  'LSTM', 'Keras', 'RNN']])}
  linear_prediction, _, _ = tf.compat.v1.feature_column.linear_model(features,
  columns)
  # or
  import tensorflow as tf
  keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords",
  10000)
  keywords_embedded = tf.feature_column.embedding_column(keywords, 16)
  columns = [keywords_embedded]
  features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
  'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
  'LSTM', 'Keras', 'RNN']])}
  input_layer = tf.keras.layers.DenseFeatures(columns)
  dense_tensor = input_layer(features)
  ```
  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.
  Returns:
    A `HashedCategoricalColumn`.
  Raises:
    ValueError: `hash_bucket_size` is not greater than 1.
    ValueError: `dtype` is neither string nor integer.
  """
  if hash_bucket_size is None:
    raise ValueError('hash_bucket_size must be set. ' 'key: {}'.format(key))
  if hash_bucket_size < 1:
    raise ValueError('hash_bucket_size must be at least 1. '
                     'hash_bucket_size: {}, key: {}'.format(
                         hash_bucket_size, key))
  fc_utils.assert_key_is_string(key)
  fc_utils.assert_string_or_int(dtype, prefix='column_name: {}'.format(key))
  return HashedCategoricalColumn(key, hash_bucket_size, dtype)
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(v1=['feature_column.categorical_column_with_vocabulary_file'])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def categorical_column_with_vocabulary_file(key,
                                            vocabulary_file,
                                            vocabulary_size=None,
                                            num_oov_buckets=0,
                                            default_value=None,
                                            dtype=dtypes.string):
  """A `CategoricalColumn` with a vocabulary file.
  Use this when your inputs are in string or integer format, and you have a
  vocabulary file that maps each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.
  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.
  Example with `num_oov_buckets`:
  File '/us/states.txt' contains 50 lines, each with a 2-character U.S. state
  abbreviation. All inputs with values in that file are assigned an ID 0-49,
  corresponding to its line number. All other values are hashed and assigned an
  ID 50-54.
  ```python
  import tensorflow as tf
  states = tf.feature_column.categorical_column_with_vocabulary_file(
    key='states', vocabulary_file='states.txt', vocabulary_size=5,
    num_oov_buckets=1)
  columns = [states]
  features = {'states':tf.constant([['california', 'georgia', 'michigan',
  'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
  'texas']])}
  linear_prediction = tf.compat.v1.feature_column.linear_model(features,
  columns)
  ```
  Example with `default_value`:
  File '/us/states.txt' contains 51 lines - the first line is 'XX', and the
  other 50 each have a 2-character U.S. state abbreviation. Both a literal 'XX'
  in input, and other values missing from the file, will be assigned ID 0. All
  others are assigned the corresponding line number 1-50.
  ```python
  import tensorflow as tf
  states = tf.feature_column.categorical_column_with_vocabulary_file(
    key='states', vocabulary_file='states.txt', vocabulary_size=6,
    default_value=0)
  columns = [states]
  features = {'states':tf.constant([['california', 'georgia', 'michigan',
  'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
  'texas']])}
  linear_prediction = tf.compat.v1.feature_column.linear_model(features,
  columns)
  ```
  And to make an embedding with either:
  ```python
  import tensorflow as tf
  states = tf.feature_column.categorical_column_with_vocabulary_file(
    key='states', vocabulary_file='states.txt', vocabulary_size=5,
    num_oov_buckets=1)
  columns = [tf.feature_column.embedding_column(states, 3)]
  features = {'states':tf.constant([['california', 'georgia', 'michigan',
  'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
  'texas']])}
  input_layer = tf.keras.layers.DenseFeatures(columns)
  dense_tensor = input_layer(features)
  ```
  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    vocabulary_file: The vocabulary file name.
    vocabulary_size: Number of the elements in the vocabulary. This must be no
      greater than length of `vocabulary_file`, if less than length, later
      values are ignored. If None, it is set to the length of `vocabulary_file`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
      the input value. A positive `num_oov_buckets` can not be specified with
      `default_value`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    dtype: The type of features. Only string and integer types are supported.
  Returns:
    A `CategoricalColumn` with a vocabulary file.
  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  """
  return categorical_column_with_vocabulary_file_v2(key, vocabulary_file,
                                                    vocabulary_size, dtype,
                                                    default_value,
                                                    num_oov_buckets)
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
