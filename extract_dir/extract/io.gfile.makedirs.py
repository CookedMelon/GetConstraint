@tf_export("io.gfile.makedirs")
def recursive_create_dir_v2(path):
  """Creates a directory and all parent/intermediate directories.
  It succeeds if path already exists and is writable.
  Args:
    path: string, name of the directory to be created
  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.RecursivelyCreateDir(compat.path_to_bytes(path))
