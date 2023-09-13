@keras_export("keras.utils.is_interactive_logging_enabled")
def is_interactive_logging_enabled():
    """Check if interactive logging is enabled.
    To switch between writing logs to stdout and `absl.logging`, you may use
    `keras.utils.enable_interactive_logging()` and
    `keras.utils.disable_interactie_logging()`.
    Returns:
      Boolean (True if interactive logging is enabled and False otherwise).
    """
    # Use `getattr` in case `INTERACTIVE_LOGGING`
    # does not have the `enable` attribute.
    return getattr(
        INTERACTIVE_LOGGING, "enable", keras_logging.INTERACTIVE_LOGGING_DEFAULT
    )
