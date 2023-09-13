"/home/cc/Workspace/tfconstraint/keras/initializers/initializers_v1.py"
@keras_export(
    v1=[
        "keras.initializers.TruncatedNormal",
        "keras.initializers.truncated_normal",
    ]
)
class TruncatedNormal(tf.compat.v1.truncated_normal_initializer):
    """Initializer that generates a truncated normal distribution.
    These values are similar to values from a `random_normal_initializer`
    except that values more than two standard deviations from the mean
    are discarded and re-drawn. This is the recommended initializer for
    neural network weights and filters.
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values to
        generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed` for behavior.
      dtype: Default data type, used if no `dtype` argument is provided when
        calling the initializer. Only floating point types are supported.
    @compatibility(TF2)
    Although it is a legacy compat.v1 api,
    `tf.compat.v1.keras.initializers.TruncatedNormal` is compatible with eager
    execution and `tf.function`.
    To switch to native TF2, switch to using
    `tf.keras.initializers.TruncatedNormal` (not from `compat.v1`) and
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
    initializer = tf.compat.v1.keras.initializers.TruncatedNormal(
      mean=mean,
      stddev=stddev,
      seed=seed,
      dtype=dtype)
    weight_one = tf.Variable(initializer(shape_one))
    weight_two = tf.Variable(initializer(shape_two))
    ```
    After:
    ```python
    initializer = tf.keras.initializers.TruncatedNormal(
      mean=mean,
      # seed=seed,  # Setting a seed in the native TF2 API
                    # causes it to produce the same initializations
                    # across multiple calls of the same initializer.
      stddev=stddev)
    weight_one = tf.Variable(initializer(shape_one, dtype=dtype))
    weight_two = tf.Variable(initializer(shape_two, dtype=dtype))
    ```
    #### How to Map Arguments
    | TF1 Arg Name      | TF2 Arg Name    | Note                       |
    | :---------------- | :-------------- | :------------------------- |
    | `mean`            | `mean`          | No change to defaults |
    | `stddev`          | `stddev`        | No change to defaults |
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
    >>> initializer = tf.compat.v1.keras.initializers.TruncatedNormal(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=False>
    After:
    >>> initializer = tf.keras.initializers.TruncatedNormal(seed=10)
    >>> a = initializer(shape=(2, 2))
    >>> b = initializer(shape=(2, 2))
    >>> tf.reduce_sum(a - b) == 0
    <tf.Tensor: shape=(), dtype=bool, numpy=True>
    @end_compatibility
    """
    def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=tf.float32):
        """Initializer that generates a truncated normal distribution.
        Args:
          mean: a python scalar or a scalar tensor. Mean of the random values to
            generate.
          stddev: a python scalar or a scalar tensor. Standard deviation of the
            random values to generate.
          seed: A Python integer. Used to create random seeds. See
            `tf.compat.v1.set_random_seed` for behavior.
          dtype: Default data type, used if no `dtype` argument is provided when
            calling the initializer. Only floating point types are supported.
        """
        super().__init__(mean=mean, stddev=stddev, seed=seed, dtype=dtype)
@keras_export(v1=["keras.initializers.lecun_normal"])
class LecunNormal(tf.compat.v1.variance_scaling_initializer):
    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )
    def get_config(self):
        return {"seed": self.seed}
@keras_export(v1=["keras.initializers.lecun_uniform"])
class LecunUniform(tf.compat.v1.variance_scaling_initializer):
    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="uniform", seed=seed
        )
    def get_config(self):
        return {"seed": self.seed}
@keras_export(v1=["keras.initializers.he_normal"])
class HeNormal(tf.compat.v1.variance_scaling_initializer):
    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )
    def get_config(self):
        return {"seed": self.seed}
@keras_export(v1=["keras.initializers.he_uniform"])
class HeUniform(tf.compat.v1.variance_scaling_initializer):
    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="uniform", seed=seed
        )
    def get_config(self):
        return {"seed": self.seed}
