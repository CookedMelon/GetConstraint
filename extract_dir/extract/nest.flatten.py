@tf_export("nest.flatten")
def flatten(structure, expand_composites=False):
  """Returns a flat list from a given structure.
  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure.
  If the structure is an atom, then returns a single-item list: [structure].
  This is the inverse of the `nest.pack_sequence_as` method that takes in a
  flattened list and re-packs it into the nested structure.
  In the case of dict instances, the sequence consists of the values, sorted by
  key to ensure deterministic behavior. This is true also for OrderedDict
  instances: their sequence order is ignored, the sorting order of keys is used
  instead. The same convention is followed in `nest.pack_sequence_as`. This
  correctly repacks dicts and OrderedDicts after they have been flattened, and
  also allows flattening an OrderedDict and then repacking it back using a
  corresponding plain dict, or vice-versa. Dictionaries with non-sortable keys
  cannot be flattened.
  Users must not modify any collections used in nest while this function is
  running.
  Examples:
  1. Python dict (ordered by key):
    >>> dict = { "key3": "value3", "key1": "value1", "key2": "value2" }
    >>> tf.nest.flatten(dict)
    ['value1', 'value2', 'value3']
  2. For a nested python tuple:
    >>> tuple = ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)
    >>> tf.nest.flatten(tuple)
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  3. For a nested dictionary of dictionaries:
    >>> dict = { "key3": {"c": (1.0, 2.0), "a": (3.0)},
    ... "key1": {"m": "val1", "g": "val2"} }
    >>> tf.nest.flatten(dict)
    ['val2', 'val1', 3.0, 1.0, 2.0]
  4. Numpy array (will not flatten):
    >>> array = np.array([[1, 2], [3, 4]])
    >>> tf.nest.flatten(array)
        [array([[1, 2],
                [3, 4]])]
  5. `tf.Tensor` (will not flatten):
    >>> tensor = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> tf.nest.flatten(tensor)
        [<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
          array([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]], dtype=float32)>]
  6. `tf.RaggedTensor`: This is a composite tensor thats representation consists
  of a flattened list of 'values' and a list of 'row_splits' which indicate how
  to chop up the flattened list into different rows. For more details on
  `tf.RaggedTensor`, please visit
  https://www.tensorflow.org/api_docs/python/tf/RaggedTensor.
  with `expand_composites=False`, we just return the RaggedTensor as is.
    >>> tensor = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2]])
    >>> tf.nest.flatten(tensor, expand_composites=False)
    [<tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2]]>]
  with `expand_composites=True`, we return the component Tensors that make up
  the RaggedTensor representation (the values and row_splits tensors)
    >>> tensor = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2]])
    >>> tf.nest.flatten(tensor, expand_composites=True)
    [<tf.Tensor: shape=(7,), dtype=int32, numpy=array([3, 1, 4, 1, 5, 9, 2],
                                                      dtype=int32)>,
     <tf.Tensor: shape=(4,), dtype=int64, numpy=array([0, 4, 4, 7])>]
  Args:
    structure: an atom or a nested structure. Note, numpy arrays are considered
      atoms and are not flattened.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.
  Returns:
    A Python list, the flattened version of the input.
  Raises:
    TypeError: The nest is or contains a dict with non-sortable keys.
  """
  return nest_util.flatten(
      nest_util.Modality.CORE, structure, expand_composites
  )
