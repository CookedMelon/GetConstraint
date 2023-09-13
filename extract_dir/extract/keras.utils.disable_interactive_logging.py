@keras_export("keras.utils.disable_interactive_logging")
def disable_interactive_logging():
    """Turn off interactive logging.
    When interactive logging is disabled, Keras sends logs to `absl.logging`.
    This is the best option when using Keras in a non-interactive
    way, such as running a training or inference job on a server.
    """
    INTERACTIVE_LOGGING.enable = False
