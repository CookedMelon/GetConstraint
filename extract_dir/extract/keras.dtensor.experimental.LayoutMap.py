@keras_export("keras.dtensor.experimental.LayoutMap", v1=[])
class LayoutMap(collections.abc.MutableMapping):
    """A dict-like object that maps string to `Layout` instances.
    `LayoutMap` uses a string as key and a `Layout` as value. There is a
    behavior difference between a normal Python dict and this class. The string
    key will be treated as a regex when retrieving the value. See the docstring
    of `get` for more details.
    See below for a usage example. You can define the naming schema
    of the `Layout`, and then retrieve the corresponding `Layout` instance.
    To use the `LayoutMap` with a `Model`, please see the docstring of
    `tf.keras.dtensor.experimental.layout_map_scope`.
    ```python
    map = LayoutMap(mesh=None)
    map['.*dense.*kernel'] = layout_2d
    map['.*dense.*bias'] = layout_1d
    map['.*conv2d.*kernel'] = layout_4d
    map['.*conv2d.*bias'] = layout_1d
    layout_1 = map['dense_1.kernel']    #   layout_1 == layout_2d
    layout_2 = map['dense_1.bias']      #   layout_2 == layout_1d
    layout_3 = map['dense_2.kernel']    #   layout_3 == layout_2d
    layout_4 = map['dense_2.bias']      #   layout_4 == layout_1d
    layout_5 = map['my_model/conv2d_123/kernel']    #   layout_5 == layout_4d
    layout_6 = map['my_model/conv2d_123/bias']      #   layout_6 == layout_1d
    ```
    Args:
      mesh: An optional `Mesh` that can be used to create all replicated
        layout as default when there isn't a layout found based on the input
        string query.
    """
    def __init__(self, mesh=None):
        self._layout_map = collections.OrderedDict()
        self._default_mesh = mesh
    def __getitem__(self, key):
        """Retrieve the corresponding layout by the string key.
        When there isn't an exact match, all the existing keys in the layout map
        will be treated as a regex and map against the input key again. The
        first match will be returned, based on the key insertion order. Return
        None if there isn't any match found.
        Args:
          key: the string key as the query for the layout.
        Returns:
          Corresponding layout based on the query.
        """
        if key in self._layout_map:
            return self._layout_map[key]
        for k in self._layout_map:
            if re.match(k, key):
                return self._layout_map[k]
        return None
    def __setitem__(self, key, layout):
        if key in self._layout_map:
            raise ValueError(
                f"{key} already exist in the LayoutMap with "
                f"value {self._layout_map[key]}. Please make sure to "
                "not use duplicated keys."
            )
        if not isinstance(layout, dtensor.Layout):
            raise ValueError(
                f"{layout} should be a dtensor.Layout type, got {type(layout)}"
            )
        self._layout_map[key] = layout
    def __delitem__(self, key):
        # let the dict to handle the key missing error
        return self._layout_map.pop(key)
    def __len__(self):
        return len(self._layout_map)
    def __iter__(self):
        return iter(self._layout_map)
    def get_default_mesh(self):
        """Return the default `Mesh` set at instance creation.
        The `Mesh` can be used to create default replicated `Layout` when there
        isn't a match of the input string query.
        """
        return self._default_mesh
    def scope(self):
        """Apply layout to all `tf.Variable` instances created under the scope.
        All `tf.Variable` instances created under this scope
        will be lazily initialized first. Once they are attached as the model
        or layer attributes, and there is a stable layout mapping for it, the
        variables will be reinitialized into a
        `tf.experimental.dtensor.DVariable` with corresponding layout.
        Note that the layout mapping will use object/attribute names as the
        keys to map the variable to the layout.
        For subclassed models, the full object/attribute name is used as the
        key. For Functional/Sequential models, we use `layer.name` as
        the key for the layer, followed by the attribute name. Keras ensures
        name uniqueness among the layers within a Functional/Sequential model.
        See the following examples that show variable object names
        for different Keras model types:
        ```python
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        layout_map['d1.kernel'] = layout_1
        layout_map['d1.bias'] = layout_2
        layout_map['d2.kernel'] = layout_3
        layout_map['d2.bias'] = layout_4
        ## Subclassed model
        class SubclassModel(tf.keras.Model):
          def __init__(self, name=None):
            super().__init__(name=name)
            self.d1 = tf.keras.layers.Dense(1000)
            self.d2 = tf.keras.layers.Dense(1000)
          def call(self, inputs):
            x = self.d1(inputs)
            return self.d2(x)
        with layout_map.scope():
          model = SubclassModel()
        inputs = tf.zeros((10, 10))
        results = model(inputs)
        model.d1.kernel.layout == layout_1
        model.d1.bias.layout == layout_2
        model.d2.kernel.layout == layout_3
        model.d2.bias.layout == layout_4
        ## Functional model
        with layout_map.scope():
          inputs = tf.keras.Input((10,), batch_size=10)
          x = tf.keras.layers.Dense(20, name='d1')(inputs)
          output = tf.keras.layers.Dense(30, name='d2')(x)
          model = tf.keras.Model(inputs, output)
        d1 = model.layers[1]
        d2 = model.layers[2]
        d1.kernel.layout == layout_1
        d1.bias.layout == layout_2
        d1.kernel.layout == layout_3
        d1.bias.layout == layout_4
        ## Sequential model
        with layout_map.scope():
          model = tf.keras.Sequential([
              tf.keras.layers.Dense(20, name='d1', input_shape=(10,)),
              tf.keras.layers.Dense(30, name='d2')
          ])
        d1 = model.layers[0]
        d2 = model.layers[1]
        d1.kernel.layout == layout_1
        d1.bias.layout == layout_2
        d1.kernel.layout == layout_3
        d1.bias.layout == layout_4
        ```
        Returns:
          A context that will lazily initialize all `tf.Variable` objects
          within the model, with their attributed layouts.
        """
        return layout_map_scope(self)
