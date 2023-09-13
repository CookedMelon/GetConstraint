@tf_export("nest.assert_same_structure")
def assert_same_structure(nest1, nest2, check_types=True,
                          expand_composites=False):
  """Asserts that two structures are nested in the same way.
  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure.
  Note the method does not check the types of atoms inside the structures.
  Examples:
  * These atom vs. atom comparisons will pass:
    >>> tf.nest.assert_same_structure(1.5, tf.Variable(1, tf.uint32))
    >>> tf.nest.assert_same_structure("abc", np.array([1, 2]))
  * These nested structure vs. nested structure comparisons will pass:
    >>> structure1 = (((1, 2), 3), 4, (5, 6))
    >>> structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    >>> structure3 = [(("a", "b"), "c"), "d", ["e", "f"]]
    >>> tf.nest.assert_same_structure(structure1, structure2)
    >>> tf.nest.assert_same_structure(structure1, structure3, check_types=False)
    >>> import collections
    >>> tf.nest.assert_same_structure(
    ...     collections.namedtuple("bar", "a b")(1, 2),
    ...     collections.namedtuple("foo", "a b")(2, 3),
    ...     check_types=False)
    >>> tf.nest.assert_same_structure(
    ...     collections.namedtuple("bar", "a b")(1, 2),
    ...     { "a": 1, "b": 2 },
    ...     check_types=False)
    >>> tf.nest.assert_same_structure(
    ...     { "a": 1, "b": 2, "c": 3 },
    ...     { "c": 6, "b": 5, "a": 4 })
    >>> ragged_tensor1 = tf.RaggedTensor.from_row_splits(
    ...       values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...       row_splits=[0, 4, 4, 7, 8, 8])
    >>> ragged_tensor2 = tf.RaggedTensor.from_row_splits(
    ...       values=[3, 1, 4],
    ...       row_splits=[0, 3])
    >>> tf.nest.assert_same_structure(
    ...       ragged_tensor1,
    ...       ragged_tensor2,
    ...       expand_composites=True)
  * These examples will raise exceptions:
    >>> tf.nest.assert_same_structure([0, 1], np.array([0, 1]))
    Traceback (most recent call last):
    ...
    ValueError: The two structures don't have the same nested structure
    >>> tf.nest.assert_same_structure(
    ...       collections.namedtuple('bar', 'a b')(1, 2),
    ...       collections.namedtuple('foo', 'a b')(2, 3))
    Traceback (most recent call last):
    ...
    TypeError: The two structures don't have the same nested structure
  Args:
    nest1: an atom or a nested structure.
    nest2: an atom or a nested structure.
    check_types: if `True` (default) types of structures are checked as well,
      including the keys of dictionaries. If set to `False`, for example a list
      and a tuple of objects will look the same if they have the same size. Note
      that namedtuples with identical name and fields are always considered to
      have the same shallow structure. Two types will also be considered the
      same if they are both list subtypes (which allows "list" and
      "_ListWrapper" from trackable dependency tracking to compare equal).
      `check_types=True` only checks type of sub-structures. The types of atoms
      are not checked.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.
  Raises:
    ValueError: If the two structures do not have the same number of atoms or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures. Only possible if `check_types` is `True`.
  """
  nest_util.assert_same_structure(
      nest_util.Modality.CORE, nest1, nest2, check_types, expand_composites
  )
