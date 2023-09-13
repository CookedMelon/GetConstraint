"/home/cc/Workspace/tfconstraint/python/keras/losses.py"
@keras_export(
    'keras.losses.cosine_similarity',
    v1=[
        'keras.metrics.cosine_proximity',
        'keras.metrics.cosine',
        'keras.losses.cosine_proximity',
        'keras.losses.cosine',
        'keras.losses.cosine_similarity',
    ])
@dispatch.add_dispatch_support
def cosine_similarity(y_true, y_pred, axis=-1):
  """Computes the cosine similarity between labels and predictions.
  Note that it is a number between -1 and 1. When it is a negative number
  between -1 and 0, 0 indicates orthogonality and values closer to -1
  indicate greater similarity. The values closer to 1 indicate greater
  dissimilarity. This makes it usable as a loss function in a setting
  where you try to maximize the proximity between predictions and
  targets. If either `y_true` or `y_pred` is a zero vector, cosine
  similarity will be 0 regardless of the proximity between predictions
  and targets.
  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`
  Standalone usage:
  >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
  >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
  >>> loss = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=1)
  >>> loss.numpy()
  array([-0., -0.999, 0.999], dtype=float32)
  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.
    axis: Axis along which to determine similarity.
  Returns:
    Cosine similarity tensor.
  """
  y_true = nn.l2_normalize(y_true, axis=axis)
  y_pred = nn.l2_normalize(y_pred, axis=axis)
  return -math_ops.reduce_sum(y_true * y_pred, axis=axis)
@keras_export('keras.losses.CosineSimilarity')
class CosineSimilarity(LossFunctionWrapper):
  """Computes the cosine similarity between labels and predictions.
  Note that it is a number between -1 and 1. When it is a negative number
  between -1 and 0, 0 indicates orthogonality and values closer to -1
  indicate greater similarity. The values closer to 1 indicate greater
  dissimilarity. This makes it usable as a loss function in a setting
  where you try to maximize the proximity between predictions and targets.
  If either `y_true` or `y_pred` is a zero vector, cosine similarity will be 0
  regardless of the proximity between predictions and targets.
  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`
  Standalone usage:
  >>> y_true = [[0., 1.], [1., 1.]]
  >>> y_pred = [[1., 0.], [1., 1.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
  >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
  >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
  >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
  >>> # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
  >>> #       = -((0. + 0.) +  (0.5 + 0.5)) / 2
  >>> cosine_loss(y_true, y_pred).numpy()
  -0.5
  >>> # Calling with 'sample_weight'.
  >>> cosine_loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  -0.0999
  >>> # Using 'sum' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> cosine_loss(y_true, y_pred).numpy()
  -0.999
  >>> # Using 'none' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> cosine_loss(y_true, y_pred).numpy()
  array([-0., -0.999], dtype=float32)
  Usage with the `compile()` API:
  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.CosineSimilarity(axis=1))
  ```
  Args:
    axis: The axis along which the cosine similarity is computed
      (the features axis). Defaults to -1.
    reduction: Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside of
      built-in training loops such as `tf.keras` `compile` and `fit`, using
      `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. Please see this
      custom training [tutorial]
      (https://www.tensorflow.org/tutorials/distribute/custom_training) for more
        details.
    name: Optional name for the instance.
  """
  def __init__(self,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='cosine_similarity'):
    super().__init__(
        cosine_similarity, reduction=reduction, name=name, axis=axis)
# Aliases.
bce = BCE = binary_crossentropy
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence = kl_divergence
logcosh = log_cosh
huber_loss = huber
def is_categorical_crossentropy(loss):
  result = ((isinstance(loss, CategoricalCrossentropy) or
             (isinstance(loss, LossFunctionWrapper) and
              loss.fn == categorical_crossentropy) or
             (hasattr(loss, '__name__') and
              loss.__name__ == 'categorical_crossentropy') or
             (loss == 'categorical_crossentropy')))
  return result
@keras_export('keras.losses.serialize')
def serialize(loss):
  """Serializes loss function or `Loss` instance.
  Args:
    loss: A Keras `Loss` instance or a loss function.
  Returns:
    Loss configuration dictionary.
  """
  return serialize_keras_object(loss)
@keras_export('keras.losses.deserialize')
def deserialize(name, custom_objects=None):
  """Deserializes a serialized loss class/function instance.
  Args:
      name: Loss configuration.
      custom_objects: Optional dictionary mapping names (strings) to custom
        objects (classes and functions) to be considered during deserialization.
  Returns:
      A Keras `Loss` instance or a loss function.
  """
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='loss function')
@keras_export('keras.losses.get')
def get(identifier):
  """Retrieves a Keras loss as a `function`/`Loss` class instance.
  The `identifier` may be the string name of a loss function or `Loss` class.
  >>> loss = tf.keras.losses.get("categorical_crossentropy")
  >>> type(loss)
  <class 'function'>
  >>> loss = tf.keras.losses.get("CategoricalCrossentropy")
  >>> type(loss)
  <class '...keras.losses.CategoricalCrossentropy'>
  You can also specify `config` of the loss to this function by passing dict
  containing `class_name` and `config` as an identifier. Also note that the
  `class_name` must map to a `Loss` class
  >>> identifier = {"class_name": "CategoricalCrossentropy",
  ...               "config": {"from_logits": True}}
  >>> loss = tf.keras.losses.get(identifier)
  >>> type(loss)
  <class '...keras.losses.CategoricalCrossentropy'>
  Args:
    identifier: A loss identifier. One of None or string name of a loss
      function/class or loss configuration dictionary or a loss function or a
      loss class instance.
  Returns:
    A Keras loss as a `function`/ `Loss` class instance.
  Raises:
    ValueError: If `identifier` cannot be interpreted.
  """
  if identifier is None:
    return None
  if isinstance(identifier, str):
    identifier = str(identifier)
    return deserialize(identifier)
  if isinstance(identifier, dict):
    return deserialize(identifier)
  if callable(identifier):
    return identifier
  raise ValueError(
      f'Could not interpret loss function identifier: {identifier}')
LABEL_DTYPES_FOR_LOSSES = {
    losses_impl.sparse_softmax_cross_entropy: 'int32',
    sparse_categorical_crossentropy: 'int32'
}
