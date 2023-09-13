@keras_export("keras.activations.elu")
@tf.__internal__.dispatch.add_dispatch_support
def elu(x, alpha=1.0):
    """Exponential Linear Unit.
    The exponential linear unit (ELU) with `alpha > 0` is:
    `x` if `x > 0` and
    `alpha * (exp(x) - 1)` if `x < 0`
    The ELU hyperparameter `alpha` controls the value to which an
    ELU saturates for negative net inputs. ELUs diminish the
    vanishing gradient effect.
    ELUs have negative values which pushes the mean of the activations
    closer to zero.
    Mean activations that are closer to zero enable faster learning as they
    bring the gradient closer to the natural gradient.
    ELUs saturate to a negative value when the argument gets smaller.
    Saturation means a small derivative which decreases the variation
    and the information that is propagated to the next layer.
    Example Usage:
    >>> import tensorflow as tf
    >>> model = tf.keras.Sequential()
    >>> model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='elu',
    ...          input_shape=(28, 28, 1)))
    >>> model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    >>> model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))
    >>> model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    >>> model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))
    <tensorflow.python.keras.engine.sequential.Sequential object ...>
    Args:
        x: Input tensor.
        alpha: A scalar, slope of negative section. `alpha` controls the value
          to which an ELU saturates for negative net inputs.
    Returns:
        The exponential linear unit (ELU) activation function: `x` if `x > 0`
          and `alpha * (exp(x) - 1)` if `x < 0`.
    Reference:
        - [Fast and Accurate Deep Network Learning by Exponential Linear Units
        (ELUs) (Clevert et al, 2016)](https://arxiv.org/abs/1511.07289)
    """
    return backend.elu(x, alpha)
