@tf_export("random.Algorithm", "random.experimental.Algorithm")
class Algorithm(enum.Enum):
  """A random-number-generation (RNG) algorithm.
  Many random-number generators (e.g. the `alg` argument of
  `tf.random.Generator` and `tf.random.stateless_uniform`) in TF allow
  you to choose the algorithm used to generate the (pseudo-)random
  numbers. You can set the algorithm to be one of the options below.
  * `PHILOX`: The Philox algorithm introduced in the paper ["Parallel
    Random Numbers: As Easy as 1, 2,
    3"](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
  * `THREEFRY`: The ThreeFry algorithm introduced in the paper
    ["Parallel Random Numbers: As Easy as 1, 2,
    3"](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
  * `AUTO_SELECT`: Allow TF to automatically select the algorithm
    depending on the accelerator device. Note that with this option,
    running the same TF program on different devices may result in
    different random numbers. Also note that TF may select an
    algorithm that is different from `PHILOX` and `THREEFRY`.
  """
  # The numbers here must match framework/rng_alg.h
  PHILOX = 1
  THREEFRY = 2
  AUTO_SELECT = 3
