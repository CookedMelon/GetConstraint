@keras_export("keras.activations.selu")
@tf.__internal__.dispatch.add_dispatch_support
def selu(x):
    """Scaled Exponential Linear Unit (SELU).
    The Scaled Exponential Linear Unit (SELU) activation function is defined as:
    - `if x > 0: return scale * x`
    - `if x < 0: return scale * alpha * (exp(x) - 1)`
    where `alpha` and `scale` are pre-defined constants
    (`alpha=1.67326324` and `scale=1.05070098`).
    Basically, the SELU activation function multiplies `scale` (> 1) with the
    output of the `tf.keras.activations.elu` function to ensure a slope larger
    than one for positive inputs.
    The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `tf.keras.initializers.LecunNormal` initializer)
    and the number of input units is "large enough"
    (see reference paper for more information).
    Example Usage:
    >>> num_classes = 10  # 10-class problem
    >>> model = tf.keras.Sequential()
    >>> model.add(tf.keras.layers.Dense(64, kernel_initializer='lecun_normal',
    ...                                 activation='selu'))
    >>> model.add(tf.keras.layers.Dense(32, kernel_initializer='lecun_normal',
    ...                                 activation='selu'))
    >>> model.add(tf.keras.layers.Dense(16, kernel_initializer='lecun_normal',
    ...                                 activation='selu'))
    >>> model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    Args:
        x: A tensor or variable to compute the activation function for.
    Returns:
        The scaled exponential unit activation: `scale * elu(x, alpha)`.
    Notes:
        - To be used together with the
          `tf.keras.initializers.LecunNormal` initializer.
        - To be used together with the dropout variant
          `tf.keras.layers.AlphaDropout` (not regular dropout).
    References:
        - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """
    return tf.nn.selu(x)
