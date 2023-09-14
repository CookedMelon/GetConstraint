@tf_export(
    "compat.dimension_at_index",
    v1=["dimension_at_index", "compat.dimension_at_index"])
def dimension_at_index(shape, index):
  """Compatibility utility required to allow for both V1 and V2 behavior in TF.
  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to
  coexist with the new behavior. This utility is a bridge between the two.
  If you want to retrieve the Dimension instance corresponding to a certain
  index in a TensorShape instance, use this utility, like this:
  ```
  # If you had this in your V1 code:
  dim = tensor_shape[i]
  # Use `dimension_at_index` as direct replacement compatible with both V1 & V2:
  dim = dimension_at_index(tensor_shape, i)
  # Another possibility would be this, but WARNING: it only works if the
  # tensor_shape instance has a defined rank.
  dim = tensor_shape.dims[i]  # `dims` may be None if the rank is undefined!
  # In native V2 code, we recommend instead being more explicit:
  if tensor_shape.rank is None:
    dim = Dimension(None)
  else:
    dim = tensor_shape.dims[i]
  # Being more explicit will save you from the following trap (present in V1):
  # you might do in-place modifications to `dim` and expect them to be reflected
  # in `tensor_shape[i]`, but they would not be (as the Dimension object was
  # instantiated on the fly.
  ```
  Args:
    shape: A TensorShape instance.
    index: An integer index.
  Returns:
    A dimension object.
  """
  assert isinstance(shape, TensorShape)
  if shape.rank is None:
    return Dimension(None)
  else:
    return shape.dims[index]
@tf_export(v1=["Dimension"])
class Dimension(object):
  """Represents the value of one dimension in a TensorShape.
  @compatibility(TF2)
  In TF2, members of a `TensorShape` object are integers. The `Dimension` class
  is not part of TF2's data model.
  Please refer to the [TensorShape section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/index#tensorshape) on common code
  patterns adapting Dimension objects to a TF2 syntax.
  @end_compatibility
  """
  __slots__ = ["_value"]
  def __init__(self, value):
    """Creates a new Dimension with the given value."""
    if isinstance(value, int):  # Most common case.
      if value < 0:
        raise ValueError("Dimension %d must be >= 0" % value)
      self._value = value
    elif value is None:
      self._value = None
    elif isinstance(value, Dimension):
      self._value = value._value
    else:
      try:
        # int(...) compensates for the int/long dichotomy on Python 2.X.
        # TODO(b/143206389): Remove once we fully migrate to 3.X.
        self._value = int(value.__index__())
      except AttributeError:
        raise TypeError(
            "Dimension value must be integer or None or have "
            "an __index__ method, got value '{0!r}' with type '{1!r}'".format(
                value, type(value))) from None
      if self._value < 0:
        raise ValueError("Dimension %d must be >= 0" % self._value)
  def __repr__(self):
    return "Dimension(%s)" % repr(self._value)
  def __str__(self):
    value = self._value
    return "?" if value is None else str(value)
  def __eq__(self, other):
    """Returns true if `other` has the same known value as this Dimension."""
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value == other.value
  def __ne__(self, other):
    """Returns true if `other` has a different known value from `self`."""
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value != other.value
  def __bool__(self):
    """Equivalent to `bool(self.value)`."""
    return bool(self._value)
  def __int__(self):
    return self._value
  # This is needed for Windows.
  # See https://github.com/tensorflow/tensorflow/pull/9780
  def __long__(self):
    return self._value
  def __index__(self):
    # Allow use in Python 3 range
    return self._value
  @property
  def value(self):
    """The value of this dimension, or None if it is unknown."""
    return self._value
  # TODO(b/225058047): Reconsider semantics.
  def is_compatible_with(self, other):
    """Returns true if `other` is compatible with this Dimension.
    Two known Dimensions are compatible if they have the same value.
    An unknown Dimension is compatible with all other Dimensions.
    Args:
      other: Another Dimension.
    Returns:
      True if this Dimension and `other` are compatible.
    """
    other = as_dimension(other)
    return (self._value is None or other.value is None or
            self._value == other.value)
  def assert_is_compatible_with(self, other):
    """Raises an exception if `other` is not compatible with this Dimension.
    Args:
      other: Another Dimension.
    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    """
    if not self.is_compatible_with(other):
      raise ValueError("Dimensions %s and %s are not compatible" %
                       (self, other))
  def merge_with(self, other):
    """Returns a Dimension that combines the information in `self` and `other`.
    Dimensions are combined as follows:
    ```python
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(None))  ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    # equivalent to tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(None))
    # raises ValueError for n != m
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(m))
    ```
    Args:
      other: Another Dimension.
    Returns:
      A Dimension containing the combined information of `self` and
      `other`.
    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    """
    other = as_dimension(other)
    self.assert_is_compatible_with(other)
    if self._value is None:
      return Dimension(other.value)
    else:
      return Dimension(self._value)
  def __add__(self, other):
    """Returns the sum of `self` and `other`.
    Dimensions are summed as follows:
    ```python
    tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m + n)
    tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value + other.value)
  def __radd__(self, other):
    """Returns the sum of `other` and `self`.
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    return self + other
  def __sub__(self, other):
    """Returns the subtraction of `other` from `self`.
    Dimensions are subtracted as follows:
    ```python
    tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m - n)
    tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is the subtraction of `other` from `self`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value - other.value)
  def __rsub__(self, other):
    """Returns the subtraction of `self` from `other`.
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is the subtraction of `self` from `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value - self._value)
  def __mul__(self, other):
    """Returns the product of `self` and `other`.
    Dimensions are summed as follows:
    ```python
    tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m * n)
    tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is the product of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value * other.value)
  def __rmul__(self, other):
    """Returns the product of `self` and `other`.
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is the product of `self` and `other`.
    """
    return self * other
  def __floordiv__(self, other):
    """Returns the quotient of `self` and `other` rounded down.
    Dimensions are divided as follows:
    ```python
    tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m // n)
    tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value // other.value)
  def __rfloordiv__(self, other):
    """Returns the quotient of `other` and `self` rounded down.
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value // self._value)
  def __div__(self, other):
    """DEPRECATED: Use `__floordiv__` via `x // y` instead.
    This function exists only for backwards compatibility purposes; new code
    should use `__floordiv__` via the syntax `x // y`.  Using `x // y`
    communicates clearly that the result rounds down, and is forward compatible
    to Python 3.
    Args:
      other: Another `Dimension`.
    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    return self // other
  def __rdiv__(self, other):
    """Use `__floordiv__` via `x // y` instead.
    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
    this function will explicitly call for usage of `//` instead.
    Args:
      other: Another `Dimension`.
    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', "
                    "please use // instead".format(type(other).__name__))
  def __truediv__(self, other):
    """Use `__floordiv__` via `x // y` instead.
    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'Dimension' and 'int'`,
    this function will explicitly call for usage of `//` instead.
    Args:
      other: Another `Dimension`.
    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: 'Dimension' and '{}', "
                    "please use // instead".format(type(other).__name__))
  def __rtruediv__(self, other):
    """Use `__floordiv__` via `x // y` instead.
    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
    this function will explicitly call for usage of `//` instead.
    Args:
      other: Another `Dimension`.
    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', "
                    "please use // instead".format(type(other).__name__))
  def __mod__(self, other):
    """Returns `self` modulo `other`.
    Dimension modulo are computed as follows:
    ```python
    tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m % n)
    tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is `self` modulo `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value % other.value)
  def __rmod__(self, other):
    """Returns `other` modulo `self`.
    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.
    Returns:
      A Dimension whose value is `other` modulo `self`.
    """
    other = as_dimension(other)
    return other % self
  def __lt__(self, other):
    """Returns True if `self` is known to be less than `other`.
    Dimensions are compared as follows:
    ```python
    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(n))    == (m < n)
    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(None)) == None
    ```
    Args:
      other: Another Dimension.
    Returns:
      The value of `self.value < other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value < other.value
  def __le__(self, other):
    """Returns True if `self` is known to be less than or equal to `other`.
    Dimensions are compared as follows:
    ```python
    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(n))    == (m <= n)
    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(None)) == None
    ```
    Args:
      other: Another Dimension.
    Returns:
      The value of `self.value <= other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value <= other.value
  def __gt__(self, other):
    """Returns True if `self` is known to be greater than `other`.
    Dimensions are compared as follows:
    ```python
    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(n))    == (m > n)
    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(None)) == None
    ```
    Args:
      other: Another Dimension.
    Returns:
      The value of `self.value > other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value > other.value
  def __ge__(self, other):
    """Returns True if `self` is known to be greater than or equal to `other`.
    Dimensions are compared as follows:
    ```python
    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(n))    == (m >= n)
    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(None)) == None
    ```
    Args:
      other: Another Dimension.
    Returns:
      The value of `self.value >= other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value >= other.value
  def __reduce__(self):
    return Dimension, (self._value,)
def as_dimension(value):
  """Converts the given value to a Dimension.
  A Dimension input will be returned unmodified.
  An input of `None` will be converted to an unknown Dimension.
  An integer input will be converted to a Dimension with that value.
  Args:
    value: The value to be converted.
  Returns:
    A Dimension corresponding to the given value.
  """
  if isinstance(value, Dimension):
    return value
  else:
    return Dimension(value)
