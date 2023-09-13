@tf_export("lookup.TextFileIndex")
class TextFileIndex:
  """The key and value content to get from each line.
  This class defines the key and value used for `tf.lookup.TextFileInitializer`.
  The key and value content to get from each line is specified either
  by the following, or a value `>=0`.
  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.
  A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.
  """
  WHOLE_LINE = -2
  LINE_NUMBER = -1
