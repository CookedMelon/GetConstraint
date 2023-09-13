@tf_export("debugging.experimental.disable_dump_debug_info")
def disable_dump_debug_info():
  """Disable the currently-enabled debugging dumping.
  If the `enable_dump_debug_info()` method under the same Python namespace
  has been invoked before, calling this method disables it. If no call to
  `enable_dump_debug_info()` has been made, calling this method is a no-op.
  Calling this method more than once is idempotent.
  """
  if hasattr(_state, "dumping_callback"):
    dump_root = _state.dumping_callback.dump_root
    tfdbg_run_id = _state.dumping_callback.tfdbg_run_id
    debug_events_writer.DebugEventsWriter(dump_root, tfdbg_run_id).Close()
    op_callbacks.remove_op_callback(_state.dumping_callback.callback)
    function_lib.remove_function_callback(
        _state.dumping_callback.function_callback)
    delattr(_state, "dumping_callback")
    logging.info("Disabled dumping callback in thread %s (dump root: %s)",
                 threading.current_thread().name, dump_root)
