@tf_export("io.gfile.rmtree")
def delete_recursively_v2(path):
  """Deletes everything under path recursively.
  Args:
    path: string, a path
  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.DeleteRecursively(compat.path_to_bytes(path))
