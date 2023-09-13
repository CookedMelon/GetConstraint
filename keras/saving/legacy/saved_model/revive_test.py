# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests reviving models from config and SavedModel.

These tests ensure that a model revived from a combination of config and
SavedModel have the expected structure.
"""

# TODO(kathywu): Move relevant tests from saved_model_test to
import shutil

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras import backend
from keras.saving.legacy.saved_model import load as keras_load
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import CustomObjectScope


class SubclassedModelNoConfig(keras.Model):
    def __init__(self, a, b):
        super().__init__()

        self.a = a
        self.b = b
        self.shared = CustomLayerNoConfig(a, b)
        self.all_layers = []

    def build(self, input_shape):
        self.all_layers.extend(
            [
                self.shared,
                CustomLayerWithConfig(self.a + 1, self.b + 2),
                CustomLayerNoConfig(self.a + 3, self.b + 4),
                keras.Sequential(
                    [
                        # TODO(b/145029112): Bug with losses when there are
                        # shared layers.  self.shared,  <-- Enable when bug is
                        # fixed.
                        CustomLayerNoConfig(self.a + 5, self.b + 6)
                    ]
                ),
            ]
        )
        super().build(input_shape)

    def call(self, inputs):
        x = inputs
        for layer in self.all_layers:
            x = layer(x)
        return x


class SparseDense(keras.layers.Dense):
    def call(self, inputs):
        input_shape = tf.stack(
            (tf.reduce_prod(tf.shape(inputs)[:-1]), self.kernel.shape[0])
        )
        output_shape = tf.concat(
            (tf.shape(inputs)[:-1], [self.kernel.shape[1]]), -1
        )
        x = tf.sparse.reshape(inputs, input_shape)
        return tf.reshape(
            self.activation(
                tf.sparse.sparse_dense_matmul(x, self.kernel) + self.bias
            ),
            output_shape,
        )


class SubclassedSparseModelNoConfig(keras.Model):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.shared = CustomLayerNoConfig(a, b)
        self.all_layers = [SparseDense(4)]

    def call(self, inputs):
        x = inputs
        for layer in self.all_layers:
            x = layer(x)
        return self.shared(x + self.a)


class SubclassedModelWithConfig(SubclassedModelNoConfig):
    def get_config(self):
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomLayerNoConfig(keras.layers.Layer):
    def __init__(self, a, b, name=None):
        super().__init__(name=name)
        self.a = tf.Variable(a, name="a")
        self.b = b

        def a_regularizer():
            return self.a * 2

        self.add_loss(a_regularizer)
        self.sum_metric = keras.metrics.Sum(name="inputs_sum")
        self.unused_metric = keras.metrics.Sum(name="not_added_to_metrics")

    def build(self, input_shape):
        self.c = tf.Variable(
            tf.constant(1.0, shape=input_shape[1:]), name=self.name + "_c"
        )

    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs))
        self.add_metric(self.sum_metric(inputs))
        self.add_metric(inputs, aggregation="mean", name="mean")

        return inputs + self.c


class CustomLayerWithConfig(CustomLayerNoConfig):
    def get_config(self):
        return {"a": backend.get_value(self.a), "b": self.b, "name": self.name}


class CustomNetworkDefaultConfig(keras.Model):
    def __init__(self, num_classes, name=None):
        inputs = keras.Input((2, 3), name="inputs")
        x = keras.layers.Flatten(name="flatten")(inputs)
        y = keras.layers.Dense(num_classes, name="outputs")(x)
        super().__init__(inputs, y, name=name)


class CustomNetworkWithConfig(CustomNetworkDefaultConfig):
    def __init__(self, num_classes, name=None):
        super().__init__(num_classes, name=name)
        self._config_dict = dict(num_classes=num_classes)

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config):
        return cls(config["num_classes"], name=config.get("name"))


class CustomNetworkWithConfigName(CustomNetworkWithConfig):
    def __init__(self, num_classes, name=None):
        super().__init__(num_classes, name=name)
        self._config_dict["name"] = self.name


class UnregisteredCustomSequentialModel(keras.Sequential):
    # This class is *not* registered in the CustomObjectScope.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add(keras.layers.InputLayer(input_shape=(2, 3)))


class FunctionalSubclassModel(keras.Model):
    def __init__(self, units):
        self.units = units
        my_input = keras.Input(shape=(2, 3), name="inputs")
        dense = keras.layers.Dense(self.units, activation="relu", name="dense")
        output = dense(my_input)
        outputs = {"output": output}
        super().__init__(inputs=[my_input], outputs=outputs)

    def get_config(self):
        return {"units": self.units}


class FunctionalSubclassModelWrongConfig(FunctionalSubclassModel):
    def get_config(self):
        return {}


# The WideDeepModel, whose name conflicts with a Keras built-in model, is
# registered in these tests.
class WideDeepModel(SubclassedModelWithConfig):
    pass


class ReviveTestBase(test_combinations.TestCase):
    def setUp(self):
        super().setUp()
        self.path = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, self.path, ignore_errors=True)

    def _assert_revived_correctness(self, model, revived):
        self.assertAllEqual(model.input_names, revived.input_names)
        self.assertAllEqual(model.output_names, revived.output_names)
        if model.inputs is not None:
            self.assertTrue(
                all(
                    [
                        i.shape.as_list() == r.shape.as_list()
                        and i.dtype == r.dtype
                        for (i, r) in zip(model.inputs, revived.inputs)
                    ]
                )
            )
            self.assertTrue(
                all(
                    [
                        i.shape.as_list() == r.shape.as_list()
                        and i.dtype == r.dtype
                        for (i, r) in zip(model.outputs, revived.outputs)
                    ]
                )
            )

        self.assertAllClose(
            self.evaluate(model.weights), self.evaluate(revived.weights)
        )
        input_arr = tf.constant(np.random.random((2, 2, 3)).astype(np.float32))
        if isinstance(revived.save_spec()[0][0], tf.SparseTensorSpec):
            input_arr = tf.sparse.from_dense(input_arr)

        self.assertAllClose(model(input_arr), revived(input_arr))
        self.assertAllClose(sum(model.losses), sum(revived.losses))
        self.assertAllClose(len(model.losses), len(revived.losses))
        self.assertEqual(len(model.metrics), len(revived.metrics))
        # TODO(b/150403085): Investigate why the metric order changes when
        # running this test in tf-nightly.
        self.assertAllClose(
            sorted([m.result() for m in model.metrics]),
            sorted([m.result() for m in revived.metrics]),
        )
        model_layers = {layer.name: layer for layer in model.layers}
        revived_layers = {layer.name: layer for layer in revived.layers}
        self.assertAllEqual(model_layers.keys(), revived_layers.keys())

        for name in model_layers:
            model_layer = model_layers[name]
            revived_layer = revived_layers[name]
            self.assertEqual(model_layer.name, revived_layer.name)
            self.assertEqual(model_layer.dtype, revived_layer.dtype)
            self.assertEqual(model_layer.trainable, revived_layer.trainable)
            if "WithConfig" in type(model_layer).__name__:
                self.assertEqual(type(model_layer), type(revived_layer))
            else:
                # When loading layers from SavedModel, a new class is
                # dynamically created with the same name.
                self.assertEqual(
                    type(model_layer).__name__, type(revived_layer).__name__
                )


# These tests take a while to run, so each should run in a separate shard
# (putting them in the same TestCase resolves this).
class TestBigModelRevive(ReviveTestBase):
    @test_combinations.run_with_all_model_types
    def test_revive(self):
        input_shape = None
        if test_utils.get_model_type() == "functional":
            input_shape = (2, 3)

        layer_with_config = CustomLayerWithConfig(1.0, 2)
        layer_without_config = CustomLayerNoConfig(3.0, 4)
        subclassed_with_config = SubclassedModelWithConfig(4.0, 6.0)
        subclassed_without_config = SubclassedModelNoConfig(7.0, 8.0)

        inputs = keras.Input((2, 3))
        x = CustomLayerWithConfig(1.0, 2)(inputs)
        x = CustomLayerNoConfig(3.0, 4)(x)
        x = SubclassedModelWithConfig(4.0, 6.0)(x)
        x = SubclassedModelNoConfig(7.0, 8.0)(x)
        inner_model_functional = keras.Model(inputs, x)

        inner_model_sequential = keras.Sequential(
            [
                CustomLayerWithConfig(1.0, 2),
                CustomLayerNoConfig(3.0, 4),
                SubclassedModelWithConfig(4.0, 6.0),
                SubclassedModelNoConfig(7.0, 8.0),
            ]
        )

        class SubclassedModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.all_layers = [
                    CustomLayerWithConfig(1.0, 2),
                    CustomLayerNoConfig(3.0, 4),
                    SubclassedModelWithConfig(4.0, 6.0),
                    SubclassedModelNoConfig(7.0, 8.0),
                ]

            def call(self, inputs):
                x = inputs
                for layer in self.all_layers:
                    x = layer(x)
                return x

        inner_model_subclassed = SubclassedModel()

        layers = [
            layer_with_config,
            layer_without_config,
            subclassed_with_config,
            subclassed_without_config,
            inner_model_functional,
            inner_model_sequential,
            inner_model_subclassed,
        ]
        model = test_utils.get_model_from_layers(
            layers, input_shape=input_shape
        )
        # Run data through the Model to create save spec and weights.
        model.predict(np.ones((10, 2, 3)), batch_size=10)

        # Test that the correct checkpointed values are loaded, whether the
        # layer is created from the config or SavedModel.
        layer_with_config.c.assign(2 * layer_with_config.c)
        layer_without_config.c.assign(3 * layer_without_config.c)

        model.save(self.path, save_format="tf")
        revived = keras_load.load(self.path)
        self._assert_revived_correctness(model, revived)


class TestModelRevive(ReviveTestBase):
    def test_revive_subclassed_with_nested_model(self):
        model = SubclassedModelNoConfig(1.0, 2.0)
        # Run data through the Model to create save spec and weights.
        model.predict(np.ones((10, 2, 3)), batch_size=10)
        model.save(self.path, save_format="tf")
        revived = keras_load.load(self.path)
        self._assert_revived_correctness(model, revived)

    def test_revive_subclassed_with_sparse_model(self):
        model = SubclassedSparseModelNoConfig(1.0, 2.0)
        # Run data through the Model to create save spec and weights.
        x = tf.sparse.from_dense(np.ones((10, 2, 3), dtype=np.float32))
        model.predict(x, batch_size=10)
        model.save(self.path, save_format="tf")
        revived = keras_load.load(self.path)
        self._assert_revived_correctness(model, revived)

    def test_revive_unregistered_sequential(self):
        model = UnregisteredCustomSequentialModel()
        x = np.random.random((2, 2, 3)).astype(np.float32)
        model(x)
        model.save(self.path, save_format="tf")
        revived = keras_load.load(self.path)
        self._assert_revived_correctness(model, revived)

    def test_revive_sequential_inputs(self):
        model = keras.models.Sequential(
            [
                keras.Input((None,), dtype=tf.string),
                keras.layers.Lambda(tf.strings.lower),
            ]
        )
        model.save(self.path, save_format="tf")
        revived = keras_load.load(self.path)
        revived_layers = list(
            revived._flatten_layers(include_self=False, recursive=False)
        )
        self.assertEqual(tf.string, revived_layers[0].dtype)

    @parameterized.named_parameters(
        ("default_config", CustomNetworkDefaultConfig),
        ("with_config", CustomNetworkWithConfig),
        ("with_config_name", CustomNetworkWithConfigName),
    )
    def test_revive_network(self, model_cls):
        model = model_cls(8)
        model.save(self.path, include_optimizer=False, save_format="tf")
        revived = keras_load.load(self.path, compile=False)
        self._assert_revived_correctness(model, revived)

    def test_functional_subclass(self):
        model = FunctionalSubclassModel(32)
        model.save(self.path, save_format="tf")
        revived = keras_load.load(self.path, compile=False)
        self._assert_revived_correctness(model, revived)

    def test_functional_subclass_wrong_config(self):
        model = FunctionalSubclassModelWrongConfig(32)
        model.save(self.path, save_format="tf")
        with self.assertRaisesRegex(TypeError, "required positional arguments"):
            keras_load.load(self.path, compile=False)

    def test_load_compiled_metrics(self):
        model = test_utils.get_small_sequential_mlp(1, 3)

        # Compile with dense categorical accuracy
        model.compile("rmsprop", "mse", "acc")
        x = np.random.random((5, 10)).astype(np.float32)
        y_true = np.random.random((5, 3)).astype(np.float32)
        model.train_on_batch(x, y_true)

        model.save(self.path, include_optimizer=True, save_format="tf")
        revived = keras_load.load(self.path, compile=True)
        self.assertAllClose(
            model.test_on_batch(x, y_true), revived.test_on_batch(x, y_true)
        )

        # Compile with sparse categorical accuracy
        model.compile("rmsprop", "mse", "acc")
        y_true = np.random.randint(0, 3, (5, 1)).astype(np.float32)
        model.train_on_batch(x, y_true)
        model.save(self.path, include_optimizer=True, save_format="tf")
        revived = keras_load.load(self.path, compile=True)
        self.assertAllClose(
            model.test_on_batch(x, y_true), revived.test_on_batch(x, y_true)
        )

    def test_revived_model_has_save_spec(self):
        model = SubclassedModelWithConfig(2, 3)
        model.predict(np.random.random((5, 10)).astype(np.float32))
        model.save(self.path, save_format="tf")
        revived = keras_load.load(self.path, compile=True)
        self.assertAllEqual(
            model._get_save_spec(dynamic_batch=False),
            revived._get_save_spec(dynamic_batch=False),
        )

    def test_load_model_with_name_conflict_registered_works(self):
        model = WideDeepModel(2, 3)
        model(np.random.random((5, 10)).astype(np.float32))
        model.save(self.path, save_format="tf")
        keras_load.load(self.path, compile=True)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    with CustomObjectScope(
        {
            "CustomLayerWithConfig": CustomLayerWithConfig,
            "CustomNetworkWithConfig": CustomNetworkWithConfig,
            "CustomNetworkWithConfigName": CustomNetworkWithConfigName,
            "SubclassedModelWithConfig": SubclassedModelWithConfig,
            "FunctionalSubclassModel": FunctionalSubclassModel,
            "FunctionalSubclassModelWrongConfig": FunctionalSubclassModelWrongConfig,  # noqa: E501
            "WideDeepModel": WideDeepModel,
        }
    ):
        tf.test.main()
