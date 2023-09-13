@keras_export("keras.utils.set_random_seed", v1=[])
def set_random_seed(seed):
    """Sets all random seeds for the program (Python, NumPy, and TensorFlow).
    You can use this utility to make almost any Keras program fully
    deterministic. Some limitations apply in cases where network communications
    are involved (e.g. parameter server distribution), which creates additional
    sources of randomness, or when certain non-deterministic cuDNN ops are
    involved.
    Calling this utility is equivalent to the following:
    ```python
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ```
    Arguments:
      seed: Integer, the random seed to use.
    """
    if not isinstance(seed, int):
        raise ValueError(
            "Expected `seed` argument to be an integer. "
            f"Received: seed={seed} (of type {type(seed)})"
        )
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    backend._SEED_GENERATOR.generator = random.Random(seed)
