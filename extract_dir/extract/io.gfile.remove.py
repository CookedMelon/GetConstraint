@tf_export("io.gfile.remove")
def delete_file_v2(path):
  """Deletes the path located at 'path'.
  Args:
    path: string, a path
  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    `NotFoundError` if the path does not exist.
  """
  _pywrap_file_io.DeleteFile(compat.path_to_bytes(path))
