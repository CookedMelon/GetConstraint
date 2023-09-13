@tf_export("io.gfile.mkdir")
def create_dir_v2(path):
  """Creates a directory with the name given by `path`.
  Args:
    path: string, name of the directory to be created
  Notes: The parent directories need to exist. Use `tf.io.gfile.makedirs`
    instead if there is the possibility that the parent dirs don't exist.
  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.CreateDir(compat.path_to_bytes(path))
