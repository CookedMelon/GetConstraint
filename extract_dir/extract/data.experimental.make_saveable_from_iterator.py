@tf_export("data.experimental.make_saveable_from_iterator")
@deprecation.deprecated(
    None, "`make_saveable_from_iterator` is intended for use in TF1 with "
    "`tf.compat.v1.Saver`. In TF2, use `tf.train.Checkpoint` instead.")
