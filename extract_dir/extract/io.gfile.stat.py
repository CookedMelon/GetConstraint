@tf_export("io.gfile.stat")
def stat_v2(path):
  """Returns file statistics for a given path.
  Args:
    path: string, path to a file
  Returns:
    FileStatistics struct that contains information about the path
  Raises:
    errors.OpError: If the operation fails.
  """
  return _pywrap_file_io.Stat(compat.path_to_str(path))
