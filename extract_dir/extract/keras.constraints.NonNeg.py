@keras_export("keras.constraints.NonNeg", "keras.constraints.non_neg")
class NonNeg(Constraint):
    """Constrains the weights to be non-negative.
    Also available via the shortcut function `tf.keras.constraints.non_neg`.
    """
    def __call__(self, w):
        return w * tf.cast(tf.greater_equal(w, 0.0), backend.floatx())
