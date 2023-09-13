@tf_export("test.assert_equal_graph_def", v1=[])
def assert_equal_graph_def_v2(expected, actual):
  """Asserts that two `GraphDef`s are (mostly) the same.
  Compares two `GraphDef` protos for equality, ignoring versions and ordering of
  nodes, attrs, and control inputs.  Node names are used to match up nodes
  between the graphs, so the naming of nodes must be consistent. This function
  ignores randomized attribute values that may appear in V2 checkpoints.
  Args:
    expected: The `GraphDef` we expected.
    actual: The `GraphDef` we have.
  Raises:
    AssertionError: If the `GraphDef`s do not match.
    TypeError: If either argument is not a `GraphDef`.
  """
  assert_equal_graph_def(actual, expected, checkpoint_v2=True,
                         hash_table_shared_name=True)
