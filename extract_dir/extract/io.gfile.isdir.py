@tf_export("io.gfile.isdir")
def is_directory_v2(path):
  """Returns whether the path is a directory or not.
  Args:
    path: string, path to a potential directory
  Returns:
    True, if the path is a directory; False otherwise
  """
  try:
    return _pywrap_file_io.IsDirectory(compat.path_to_bytes(path))
  except errors.OpError:
    return False
