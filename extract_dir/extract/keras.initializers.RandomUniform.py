@keras_export(
    v1=[
        "keras.initializers.RandomUniform",
        "keras.initializers.random_uniform",
        "keras.initializers.uniform",
    ]
)
class RandomUniform(tf.compat.v1.random_uniform_initializer):
    """Initializer that generates tensors with a uniform distribution.
    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range of
        random values to generate. Defaults to `-0.05`.
      maxval: A python scalar or a scalar tensor. Upper bound of the range of
        random values to generate. Defaults to `0.05`.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed` for behavior.
      dtype: Default data type, used if no `dtype` argument is provided when
        calling the initializer.
    @compatibility(TF2)
    Although it is a legacy `compat.v1` api,
    `tf.compat.v1.keras.initializers.RandomUniform` is compatible with eager
    execution and `tf.function`.
    To switch to native TF2, switch to using
    `tf.keras.initializers.RandomUniform` (not from `compat.v1`) and
    if you need to change the default dtype use
    `tf.keras.backend.set_floatx(float_dtype)`
    or pass the dtype when calling the initializer, rather than passing it
    when constructing the initializer.
    Random seed behavior:
    Also be aware that if you pass a seed to the TF2 initializer
    API it will reuse that same seed for every single initialization
    (unlike the TF1 initializer)
    #### Structural Mapping to Native TF2
    Before:
    ```python
    initializer = tf.compat.v1.keras.initializers.RandomUniform(
      minval=minval,
      maxval=maxval,
      seed=seed,
      dtype=dtype)
    weight_one = tf.Variable(initializer(shape_one))
    weight_two = tf.Variable(initializer(shape_two))
    ```
    After:
    ```python
    initializer = tf.keras.initializers.RandomUniform(
      minval=minval,
      maxval=maxval,
      # seed=seed,  # Setting a seed in the native TF2 API
                    # causes it to produce the same initializations
                    # across multiple calls of the same initializer.
      )
    weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
    weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
    ```
    #### How to Map Arguments
    | TF1 Arg Name      | TF2 Arg Name    | Note                       |
    | :---------------- | :-------------- | :------------------------- |
    | `minval`            | `minval`          | No change to defaults |
    | `maxval`          | `maxval`        | No change to defaults |
    | `seed`            | `seed`          | Different random number generation |
    :                    :        : semantics (to change in a :
    :                    :        : future version). If set, the TF2 version :
    :                    :        : will use stateless random number :
    :                    :        : generation which will produce the exact :
    :                    :        : same initialization even across multiple :
    :                    :        : calls of the initializer instance. the :
    :                    :        : `compat.v1` version will generate new :
    :                    :        : initializations each time. Do not set :
    :                    :        : a seed if you need different          :
    :                    :        : initializations each time. Instead    :
    :                    :        : either set a global tf seed with
    :                    :        : `tf.random.set_seed` if you need :
    :                    :        : determinism, or initialize each weight :
    :                    :        : with a separate initializer instance  :
    :                    :        : and a different seed.                 :
    | `dtype`           | `dtype`  | The TF2 native api only takes it  |
    :                   :      : as a `__call__` arg, not a constructor arg. :
    | `partition_info`  | -    |  (`__call__` arg in TF1) Not supported      |
    #### Example of fixed-seed behavior differences
    `compat.v1` Fixed seed behavior:
    >>> initializer = tf.compat.v1.keras.initializers.RandomUniform(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=False>
    After:
    >>> initializer = tf.keras.initializers.RandomUniform(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=True>
    @end_compatibility
    """
    def __init__(self, minval=-0.05, maxval=0.05, seed=None, dtype=tf.float32):
        super().__init__(minval=minval, maxval=maxval, seed=seed, dtype=dtype)
