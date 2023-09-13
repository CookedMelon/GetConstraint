@tf_export("io.gfile.glob")
def get_matching_files_v2(pattern):
  r"""Returns a list of files that match the given pattern(s).
  The patterns are defined as strings. Supported patterns are defined
  here. Note that the pattern can be a Python iteratable of string patterns.
  The format definition of the pattern is:
  **pattern**: `{ term }`
  **term**:
    * `'*'`: matches any sequence of non-'/' characters
    * `'?'`: matches a single non-'/' character
    * `'[' [ '^' ] { match-list } ']'`: matches any single
      character (not) on the list
    * `c`: matches character `c`  where `c != '*', '?', '\\', '['`
    * `'\\' c`: matches character `c`
  **character range**:
    * `c`: matches character `c` while `c != '\\', '-', ']'`
    * `'\\' c`: matches character `c`
    * `lo '-' hi`: matches character `c` for `lo <= c <= hi`
  Examples:
  >>> tf.io.gfile.glob("*.py")
  ... # For example, ['__init__.py']
  >>> tf.io.gfile.glob("__init__.??")
  ... # As above
  >>> files = {"*.py"}
  >>> the_iterator = iter(files)
  >>> tf.io.gfile.glob(the_iterator)
  ... # As above
  See the C++ function `GetMatchingPaths` in
  [`core/platform/file_system.h`]
  (../../../core/platform/file_system.h)
  for implementation details.
  Args:
    pattern: string or iterable of strings. The glob pattern(s).
  Returns:
    A list of strings containing filenames that match the given pattern(s).
  Raises:
    errors.OpError: If there are filesystem / directory listing errors.
    errors.NotFoundError: If pattern to be matched is an invalid directory.
  """
  if isinstance(pattern, six.string_types):
    return [
        # Convert the filenames to string from bytes.
        compat.as_str_any(matching_filename)
        for matching_filename in _pywrap_file_io.GetMatchingFiles(
            compat.as_bytes(pattern))
    ]
  else:
    return [
        # Convert the filenames to string from bytes.
        compat.as_str_any(matching_filename)  # pylint: disable=g-complex-comprehension
        for single_filename in pattern
        for matching_filename in _pywrap_file_io.GetMatchingFiles(
            compat.as_bytes(single_filename))
    ]
