@tf_export("io.gfile.listdir")
def list_directory_v2(path):
  """Returns a list of entries contained within a directory.
  The list is in arbitrary order. It does not contain the special entries "."
  and "..".
  Args:
    path: string, path to a directory
  Returns:
    [filename1, filename2, ... filenameN] as strings
  Raises:
    errors.NotFoundError if directory doesn't exist
  """
  if not is_directory(path):
    raise errors.NotFoundError(
        node_def=None,
        op=None,
        message="Could not find directory {}".format(path))
  # Convert each element to string, since the return values of the
  # vector of string should be interpreted as strings, not bytes.
  return [
      compat.as_str_any(filename)
      for filename in _pywrap_file_io.GetChildren(compat.path_to_bytes(path))
  ]
