@tf_export("data.experimental.Reducer")
class Reducer:
  """A reducer is used for reducing a set of elements.
  A reducer is represented as a tuple of the three functions:
  - init_func - to define initial value: key => initial state
  - reducer_func - operation to perform on values with same key: (old state, input) => new state
  - finalize_func - value to return in the end: state => result
  For example,
  ```
  def init_func(_):
    return (0.0, 0.0)
  def reduce_func(state, value):
    return (state[0] + value['features'], state[1] + 1)
  def finalize_func(s, n):
    return s / n
  reducer = tf.data.experimental.Reducer(init_func, reduce_func, finalize_func)
  ```
  """
  def __init__(self, init_func, reduce_func, finalize_func):
    self._init_func = init_func
    self._reduce_func = reduce_func
    self._finalize_func = finalize_func
  @property
  def init_func(self):
    return self._init_func
  @property
  def reduce_func(self):
    return self._reduce_func
  @property
  def finalize_func(self):
    return self._finalize_func
