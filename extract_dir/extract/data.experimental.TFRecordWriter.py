@tf_export("data.experimental.TFRecordWriter")
@deprecation.deprecated(
    None, "To write TFRecords to disk, use `tf.io.TFRecordWriter`. To save "
    "and load the contents of a dataset, use `tf.data.experimental.save` "
    "and `tf.data.experimental.load`")
