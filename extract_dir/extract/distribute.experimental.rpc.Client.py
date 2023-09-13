@tf_export("distribute.experimental.rpc.Client", v1=[])
class Client(object):
  """Client class for invoking RPCs to the server."""
  @staticmethod
  def create(rpc_layer, address, name="", timeout_in_ms=0):
    """Create TF RPC client to connect to the given address.
    Args:
      rpc_layer: Communication layer between client and server. Only "grpc" rpc
        layer is supported at the moment.
      address: Address of the server to connect the RPC client to.
      name: Name of the RPC Client. You can create multiple clients connecting
        to same server and distinguish them using different names.
      timeout_in_ms: The default timeout to use for outgoing RPCs from client. 0
        indicates no timeout. Exceeding timeout during RPC will raise
        DeadlineExceeded error.
    Returns:
      An instance of `tf.distribute.experimental.rpc.Client` with the following
      dynamically added methods for eagerly created clients:
        * `Registered methods` e.g. multiply(**args):
            If Client is created when executing eagerly, client will request the
            list of registered methods from server during client creation.
            The convenience methods for RPCs will be dynamically added to the
            created Client instance.
            For example, when a server has method "multiply" registered, the
            client object created in eager mode will have 'multiply' method
            available. Users can use client.multiply(..) to make RPC, instead of
            client.call("multiply", ...)
            Both "call" and "multiply" methods are non-blocking i.e. they return
            a StatusOrResult object which should be used to wait for getting
            value or error.
            Along with the above, blocking versions of the registered
            methods are also dynamically added to client instance.
            e.g. multiply_blocking(**args). These methods block till the RPC is
            finished and return response for successful RPC. Otherwise raise
            exception.
            These methods are not available when Client is created inside a
            tf.function.
    Raises:
        A ValueError if rpc_layer other than "grpc" is used. Only GRPC
          is supported at the moment.
        A DeadlineExceeded exception in eager mode if timeout exceeds while
          creating and listing client methods.
    Example usage:
      >>> # Have server already started.
      >>> import portpicker
      >>> @tf.function(input_signature=[
      ...      tf.TensorSpec([], tf.int32),
      ...      tf.TensorSpec([], tf.int32)])
      ... def remote_fn(a, b):
      ...   return tf.add(a, b)
      >>> port = portpicker.pick_unused_port()
      >>> address = "localhost:{}".format(port)
      >>> server = tf.distribute.experimental.rpc.Server.create("grpc", address)
      >>> server.register("addition", remote_fn)
      >>> server.start()
      >>> # Start client
      >>> client = tf.distribute.experimental.rpc.Client.create("grpc",
      ...      address=address, name="test_client")
      >>> a = tf.constant(2, dtype=tf.int32)
      >>> b = tf.constant(3, dtype=tf.int32)
      >>> result = client.call(
      ...    args=[a, b],
      ...    method_name="addition",
      ...    output_specs=tf.TensorSpec((), tf.int32))
      >>> if result.is_ok():
      ...   result.get_value()
      >>> result = client.addition(a, b)
      >>> if result.is_ok():
      ...   result.get_value()
      >>> value = client.addition_blocking(a, b)
    """
    if rpc_layer != "grpc":
      raise ValueError("Only GRPC backend is supported at the moment.")
    if context.executing_eagerly():
      list_registered_methods = True
    else:
      list_registered_methods = False
    return GrpcClient(
        address=address,
        name=name,
        list_registered_methods=list_registered_methods,
        timeout_in_ms=timeout_in_ms)
  def call(self,
           method_name: str,
           args: Optional[Sequence[core_tf_types.Tensor]] = None,
           output_specs=None,
           timeout_in_ms=0):
    """Method for making RPC calls to remote server.
    This invokes RPC to the server, executing the registered method_name
    remotely.
    Args:
      method_name: Remote registered method to invoke
      args: List of arguments for the registered method.
      output_specs: Output specs for the output from method.
         For example, if tf.function is: @tf.function(input_signature=[
           tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32) ])
          def multiply_fn(a, b): return tf.math.multiply(a, b)
        output_spec is: tf.TensorSpec((), tf.int32)  If you have access to TF
          Function, the output specs can be generated
       from tf.function by calling: output_specs =
         tf.nest.map_structure(tf.type_spec_from_value,
         tf_function.get_concrete_function().structured_outputs  If output_specs
         are not provided, flattened list of tensors will be returned in
         response.
      timeout_in_ms: Timeout for this call. If 0, default client timeout will be
        used.
    Returns:
      An instance of `StatusOrResult` class with the following available
      methods.
        * `is_ok()`:
            Returns True of RPC was successful.
        * `get_error()`:
            Returns TF error_code and error message for the RPC.
        * `get_value()`:
            Returns the returned value from remote TF function execution
            when RPC is successful.
      Calling any of the above methods will block till RPC is completed and
      result is available.
    """
    raise NotImplementedError("Must be implemented in inherited classes.")
