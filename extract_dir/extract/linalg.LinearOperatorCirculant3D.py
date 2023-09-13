@tf_export("linalg.LinearOperatorCirculant3D")
@linear_operator.make_composite_tensor
class LinearOperatorCirculant3D(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a nested block circulant matrix.
  This operator acts like a block circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.
  #### Description in terms of block circulant matrices
  If `A` is nested block circulant, with block sizes `N0, N1, N2`
  (`N0 * N1 * N2 = N`):
  `A` has a block structure, composed of `N0 x N0` blocks, with each
  block an `N1 x N1` block circulant matrix.
  For example, with `W`, `X`, `Y`, `Z` each block circulant,
  ```
  A = |W Z Y X|
      |X W Z Y|
      |Y X W Z|
      |Z Y X W|
  ```
  Note that `A` itself will not in general be circulant.
  #### Description in terms of the frequency spectrum
  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.
  If `H.shape = [N0, N1, N2]`, (`N0 * N1 * N2 = N`):
  Loosely speaking, matrix multiplication is equal to the action of a
  Fourier multiplier:  `A u = IDFT3[ H DFT3[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT3[u]` be the
  `[N0, N1, N2, R]` `Tensor` defined by re-shaping `u` to `[N0, N1, N2, R]` and
  taking a three dimensional DFT across the first three dimensions.  Let `IDFT3`
  be the inverse of `DFT3`.  Matrix multiplication may be expressed columnwise:
  ```(A u)_r = IDFT3[ H * (DFT3[u])_r ]```
  #### Operator properties deduced from the spectrum.
  * This operator is positive definite if and only if `Real{H} > 0`.
  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.
  Suppose `H.shape = [B1,...,Bb, N0, N1, N2]`, we say that `H` is a Hermitian
  spectrum if, with `%` meaning modulus division,
  ```
  H[..., n0 % N0, n1 % N1, n2 % N2]
    = ComplexConjugate[ H[..., (-n0) % N0, (-n1) % N1, (-n2) % N2] ].
  ```
  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.
  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.
  ### Examples
  See `LinearOperatorCirculant` and `LinearOperatorCirculant2D` for examples.
  #### Performance
  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then
  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.
  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.
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
               spectrum,
               input_output_dtype=dtypes.complex64,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name="LinearOperatorCirculant3D"):
    """Initialize an `LinearOperatorCirculant`.
    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`
    by providing `spectrum`, a `[B1,...,Bb, N0, N1, N2]` `Tensor`
    with `N0*N1*N2 = N`.
    If `input_output_dtype = DTYPE`:
    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.
    * Values returned by all methods, such as `matmul` or `determinant` will be
      cast to `DTYPE`.
    Note that if the spectrum is not Hermitian, then this operator corresponds
    to a complex matrix with non-zero imaginary part.  In this case, setting
    `input_output_dtype` to a real type will forcibly cast the output to be
    real, resulting in incorrect results!
    If on the other hand the spectrum is Hermitian, then this operator
    corresponds to a real-valued matrix, and setting `input_output_dtype` to
    a real type is fine.
    Args:
      spectrum:  Shape `[B1,...,Bb, N0, N1, N2]` `Tensor`.  Allowed dtypes:
        `float16`, `float32`, `float64`, `complex64`, `complex128`.
        Type can be different than `input_output_dtype`
      input_output_dtype: `dtype` for input/output.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the real part of all eigenvalues is positive.  We do not require
        the operator to be self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name:  A name to prepend to all ops created by this class.
    """
    parameters = dict(
        spectrum=spectrum,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    super(LinearOperatorCirculant3D, self).__init__(
        spectrum,
        block_depth=3,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        parameters=parameters,
        name=name)
