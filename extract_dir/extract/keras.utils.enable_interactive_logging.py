@keras_export("keras.utils.enable_interactive_logging")
def enable_interactive_logging():
    """Turn on interactive logging.
    When interactive logging is enabled, Keras displays logs via stdout.
    This provides the best experience when using Keras in an interactive
    environment such as a shell or a notebook.
    """
    INTERACTIVE_LOGGING.enable = True
