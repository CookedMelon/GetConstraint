@tf_export("distribute.experimental.rpc.Server", v1=[])
class Server(object):
  """A Server base class for accepting RPCs for registered tf.functions.
    Functions can be registered on the server and are exposed via RPCs.
  """
  @staticmethod
  def create(rpc_layer, address):
    """Create TF RPC server at given address.
    Args:
      rpc_layer: Communication layer between client and server. Only "grpc" rpc
        layer is supported at the moment.
      address: Address where RPC server is hosted.
    Returns:
      An instance of `tf.distribute.experimental.rpc.Server` class.
    Raises:
        A ValueError if rpc_layer other than "grpc" is used. Only GRPC
        is supported at the moment.
    Example usage:
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
    """
    if rpc_layer != "grpc":
      raise ValueError("Only GRPC backend is supported at the moment.")
    return GrpcServer(address=address)
  def register(self, method_name: str,
               func: Union[def_function.Function,
                           tf_function.ConcreteFunction]):
    """Method for registering tf.function on server.
    Registered methods can be invoked remotely from clients.
    Args:
      method_name: Name of the tf.function. Clients use this method_name to make
        RPCs.
      func: A `tf.function` or ConcreteFunction to register.
    """
    raise NotImplementedError("Please use create_server method to create a"
                              "concrete subclass of Server.")
  def start(self):
    """Starts the RPC server on provided address.
     Server listens for new requests from client, once it is started.
    """
    raise NotImplementedError("Please use create_server method to create a"
                              "concrete subclass of Server.")
