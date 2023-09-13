@tf_export("io.gfile.rename")
def rename_v2(src, dst, overwrite=False):
  """Rename or move a file / directory.
  Args:
    src: string, pathname for a file
    dst: string, pathname to which the file needs to be moved
    overwrite: boolean, if false it's an error for `dst` to be occupied by an
      existing file.
  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.RenameFile(
      compat.path_to_bytes(src), compat.path_to_bytes(dst), overwrite)
