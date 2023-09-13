@tf_export("linalg.LinearOperatorScaledIdentity")
@linear_operator.make_composite_tensor
class LinearOperatorScaledIdentity(BaseLinearOperatorIdentity):
  """`LinearOperator` acting like a scaled [batch] identity matrix `A = c I`.
  This operator acts like a scaled [batch] identity matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  a scaled version of the `N x N` identity matrix.
  `LinearOperatorIdentity` is initialized with `num_rows`, and a `multiplier`
  (a `Tensor`) of shape `[B1,...,Bb]`.  `N` is set to `num_rows`, and the
  `multiplier` determines the scale for each batch member.
  ```python
  # Create a 2 x 2 scaled identity matrix.
  operator = LinearOperatorIdentity(num_rows=2, multiplier=3.)
  operator.to_dense()
  ==> [[3., 0.]
       [0., 3.]]
  operator.shape
  ==> [2, 2]
  operator.log_abs_determinant()
  ==> 2 * Log[3]
  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> 3 * x
  y = tf.random.normal(shape=[3, 2, 4])
  # Note that y.shape is compatible with operator.shape because operator.shape
  # is broadcast to [3, 2, 2].
  x = operator.solve(y)
  ==> 3 * x
  # Create a 2-batch of 2x2 identity matrices
  operator = LinearOperatorIdentity(num_rows=2, multiplier=5.)
  operator.to_dense()
  ==> [[[5., 0.]
        [0., 5.]],
       [[5., 0.]
        [0., 5.]]]
  x = ... Shape [2, 2, 3]
  operator.matmul(x)
  ==> 5 * x
  # Here the operator and x have different batch_shape, and are broadcast.
  x = ... Shape [1, 2, 3]
  operator.matmul(x)
  ==> 5 * x
  ```
  ### Shape compatibility
  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if
  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```
  ### Performance
  * `operator.matmul(x)` is `O(D1*...*Dd*N*R)`
  * `operator.solve(x)` is `O(D1*...*Dd*N*R)`
  * `operator.determinant()` is `O(D1*...*Dd)`
  #### Matrix property hints
  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """
  def __init__(self,
               num_rows,
               multiplier,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorScaledIdentity"):
    r"""Initialize a `LinearOperatorScaledIdentity`.
    The `LinearOperatorScaledIdentity` is initialized with `num_rows`, which
    determines the size of each identity matrix, and a `multiplier`,
    which defines `dtype`, batch shape, and scale of each matrix.
    This operator is able to broadcast the leading (batch) dimensions.
    Args:
      num_rows:  Scalar non-negative integer `Tensor`.  Number of rows in the
        corresponding identity matrix.
      multiplier:  `Tensor` of shape `[B1,...,Bb]`, or `[]` (a scalar).
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      assert_proper_shapes:  Python `bool`.  If `False`, only perform static
        checks that initialization and method arguments have proper shape.
        If `True`, and static checks are inconclusive, add asserts to the graph.
      name: A name for this `LinearOperator`
    Raises:
      ValueError:  If `num_rows` is determined statically to be non-scalar, or
        negative.
    """
    parameters = dict(
        num_rows=num_rows,
        multiplier=multiplier,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        assert_proper_shapes=assert_proper_shapes,
        name=name)
    self._assert_proper_shapes = assert_proper_shapes
    with ops.name_scope(name, values=[multiplier, num_rows]):
      self._multiplier = linear_operator_util.convert_nonref_to_tensor(
          multiplier, name="multiplier")
      # Check and auto-set hints.
      if not self._multiplier.dtype.is_complex:
        if is_self_adjoint is False:  # pylint: disable=g-bool-id-comparison
          raise ValueError("A real diagonal operator is always self adjoint.")
        else:
          is_self_adjoint = True
      if not is_square:
        raise ValueError("A ScaledIdentity operator is always square.")
      linear_operator_util.assert_not_ref_type(num_rows, "num_rows")
      super(LinearOperatorScaledIdentity, self).__init__(
          dtype=self._multiplier.dtype.base_dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)
      self._num_rows = linear_operator_util.shape_tensor(
          num_rows, name="num_rows")
      self._num_rows_static = tensor_util.constant_value(self._num_rows)
      self._check_num_rows_possibly_add_asserts()
      self._num_rows_cast_to_dtype = math_ops.cast(self._num_rows, self.dtype)
      self._num_rows_cast_to_real_dtype = math_ops.cast(self._num_rows,
                                                        self.dtype.real_dtype)
  def _shape(self):
    matrix_shape = tensor_shape.TensorShape((self._num_rows_static,
                                             self._num_rows_static))
    batch_shape = self.multiplier.shape
    return batch_shape.concatenate(matrix_shape)
  def _shape_tensor(self):
    matrix_shape = array_ops_stack.stack(
        (self._num_rows, self._num_rows), axis=0)
    batch_shape = array_ops.shape(self.multiplier)
    return array_ops.concat((batch_shape, matrix_shape), 0)
  def _assert_non_singular(self):
    return check_ops.assert_positive(
        math_ops.abs(self.multiplier), message="LinearOperator was singular")
  def _assert_positive_definite(self):
    return check_ops.assert_positive(
        math_ops.real(self.multiplier),
        message="LinearOperator was not positive definite.")
  def _assert_self_adjoint(self):
    imag_multiplier = math_ops.imag(self.multiplier)
    return check_ops.assert_equal(
        array_ops.zeros_like(imag_multiplier),
        imag_multiplier,
        message="LinearOperator was not self-adjoint")
  def _make_multiplier_matrix(self, conjugate=False):
    # Shape [B1,...Bb, 1, 1]
    multiplier_matrix = array_ops.expand_dims(
        array_ops.expand_dims(self.multiplier, -1), -1)
    if conjugate:
      multiplier_matrix = math_ops.conj(multiplier_matrix)
    return multiplier_matrix
  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x = linalg.adjoint(x) if adjoint_arg else x
    if self._assert_proper_shapes:
      aps = linear_operator_util.assert_compatible_matrix_dimensions(self, x)
      x = control_flow_ops.with_dependencies([aps], x)
    return x * self._make_multiplier_matrix(conjugate=adjoint)
  def _determinant(self):
    return self.multiplier**self._num_rows_cast_to_dtype
  def _log_abs_determinant(self):
    return self._num_rows_cast_to_real_dtype * math_ops.log(
        math_ops.abs(self.multiplier))
  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    if self._assert_proper_shapes:
      aps = linear_operator_util.assert_compatible_matrix_dimensions(self, rhs)
      rhs = control_flow_ops.with_dependencies([aps], rhs)
    return rhs / self._make_multiplier_matrix(conjugate=adjoint)
  def _trace(self):
    # Get Tensor of all ones of same shape as self.batch_shape.
    if self.batch_shape.is_fully_defined():
      batch_of_ones = array_ops.ones(shape=self.batch_shape, dtype=self.dtype)
    else:
      batch_of_ones = array_ops.ones(
          shape=self.batch_shape_tensor(), dtype=self.dtype)
    if self._min_matrix_dim() is not None:
      return self.multiplier * self._min_matrix_dim() * batch_of_ones
    else:
      return (self.multiplier * math_ops.cast(self._min_matrix_dim_tensor(),
                                              self.dtype) * batch_of_ones)
  def _diag_part(self):
    return self._ones_diag() * self.multiplier[..., array_ops.newaxis]
  def add_to_tensor(self, mat, name="add_to_tensor"):
    """Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.
    Args:
      mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
      name:  A name to give this `Op`.
    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      # Shape [B1,...,Bb, 1]
      multiplier_vector = array_ops.expand_dims(self.multiplier, -1)
      # Shape [C1,...,Cc, M, M]
      mat = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          mat, name="mat"
      )
      # Shape [C1,...,Cc, M]
      mat_diag = array_ops.matrix_diag_part(mat)
      # multiplier_vector broadcasts here.
      new_diag = multiplier_vector + mat_diag
      return array_ops.matrix_set_diag(mat, new_diag)
  def _eigvals(self):
    return self._ones_diag() * self.multiplier[..., array_ops.newaxis]
  def _cond(self):
    # Condition number for a scalar time identity matrix is one, except when the
    # scalar is zero.
    return array_ops.where_v2(
        math_ops.equal(self._multiplier, 0.),
        math_ops.cast(np.nan, dtype=self.dtype),
        math_ops.cast(1., dtype=self.dtype))
  @property
  def multiplier(self):
    """The [batch] scalar `Tensor`, `c` in `cI`."""
    return self._multiplier
  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("num_rows",)
  @property
  def _composite_tensor_fields(self):
    return ("num_rows", "multiplier", "assert_proper_shapes")
  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {"multiplier": 0}
