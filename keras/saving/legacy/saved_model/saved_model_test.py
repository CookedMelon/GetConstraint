# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for saving and loading Keras models and layers from SavedModel.

These should ensure that all layer properties are correctly assigned after
loading from the SavedModel.

Tests that focus on the model structure should go in revive_test.py
"""

import os
import shutil
import sys

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2

import keras
from keras import regularizers
from keras.feature_column.dense_features import DenseFeatures
from keras.protobuf import saved_metadata_pb2
from keras.protobuf import versions_pb2
from keras.saving import object_registration
from keras.saving.legacy.saved_model import json_utils
from keras.saving.legacy.saved_model import load as keras_load
from keras.saving.legacy.saved_model import save_impl as keras_save
from keras.saving.legacy.saved_model import utils as saved_model_utils
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import control_flow_util
from keras.utils import tf_contextlib
from keras.utils import tf_inspect


class LayerWithLearningPhase(keras.engine.base_layer.Layer):
    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(
            shape=[None] * len(input_shape)
        )
        self.built = True

    def call(self, x, training=None):
        if training is None:
            training = keras.backend.learning_phase()
        output = control_flow_util.smart_cond(
            training, lambda: x * 0, lambda: tf.identity(x)
        )
        if not tf.executing_eagerly():
            output._uses_learning_phase = True
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def _use_input_spec_as_call_signature(self):
        return True


class LayerWithLoss(keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs))
        return inputs * 2


class LayerWithUpdate(keras.layers.Layer):
    def build(self, _):
        self.v = self.add_weight(
            "v",
            shape=[],
            initializer=keras.initializers.zeros,
            trainable=False,
            dtype=tf.float32,
        )

    def call(self, inputs, training=True):
        if training:
            self.add_update(self.v.assign_add(1.0))
        return inputs * 2.0


@object_registration.register_keras_serializable("Testing")
class GlobalLayerThatShouldFailIfNotAdded(keras.layers.Layer):
    _must_restore_from_config = True


@test_combinations.run_all_keras_modes
class TestSavedModelFormatAllModes(test_combinations.TestCase):
    def _save_model_dir(self, dirname="saved_model"):
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        return os.path.join(temp_dir, dirname)

    def _get_model(self):
        model = test_utils.get_small_mlp(1, 4, input_dim=3)
        model.layers[-1].activity_regularizer = regularizers.get("l2")
        model.activity_regularizer = regularizers.get("l2")
        model.compile(loss="mse", optimizer="rmsprop")

        def callable_loss():
            return tf.reduce_sum(model.weights[0])

        model.add_loss(callable_loss)
        return model

    def _train_model(self, model, use_dataset=False):
        x = np.random.random((1, 3))
        y = np.random.random((1, 4))

        if not tf.__internal__.tf2.enabled():
            # The layer autocast behavior only runs when autocast is enabled, so
            # in V1, the numpy inputs still need to be cast to float32.
            x = x.astype(np.float32)
            y = y.astype(np.float32)

        if use_dataset:
            dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)
            model.fit(dataset)
        else:
            model.train_on_batch(x, y)

    def _save_and_load(self, model):
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)
        return loaded

    def _test_evaluation(self, model, loaded):
        # Assert that original and loaded models have the same results when
        # called.
        self.evaluate(tf.compat.v1.variables_initializer(loaded.variables))
        self.assertAllClose(
            self.evaluate(model.weights), self.evaluate(loaded.weights)
        )

        input_arr = tf.constant(np.random.random((1, 3)).astype(np.float32))
        self.assertAllClose(
            self.evaluate(model(input_arr)), self.evaluate(loaded(input_arr))
        )
        # Validate losses. The order of conditional losses may change between
        # the model and loaded model, so sort the losses first.
        if tf.executing_eagerly():
            self.assertAllClose(
                sorted(self.evaluate(model.losses)),
                sorted(self.evaluate(loaded.losses)),
            )

    @test_combinations.run_with_all_model_types
    def test_model_save_and_load(self):
        model = self._get_model()
        self._train_model(model, use_dataset=False)
        loaded = self._save_and_load(model)
        self._test_evaluation(model, loaded)

    @test_combinations.run_with_all_model_types
    def test_model_save_and_load_dataset(self):
        model = self._get_model()
        self._train_model(model, use_dataset=True)
        loaded = self._save_and_load(model)
        self._test_evaluation(model, loaded)

    def test_trainable_weights(self):
        """Tests that trainable status of individual weights is preserved."""
        layer = keras.layers.Dense(4, name="custom_layer")
        layer.build([None, 3])
        layer.add_weight(
            "extra_weight",
            shape=[],
            initializer=tf.compat.v1.constant_initializer(11),
            trainable=True,
        )
        layer.add_weight(
            "extra_weight_2",
            shape=[],
            initializer=tf.compat.v1.constant_initializer(12),
            trainable=False,
        )
        model = keras.Sequential(
            [
                keras.Input(
                    [
                        3,
                    ]
                ),
                layer,
            ]
        )

        saved_model_dir = self._save_model_dir()
        self.evaluate(tf.compat.v1.variables_initializer(layer.variables))
        model.save(saved_model_dir, save_format="tf")
        loaded_model = keras_load.load(saved_model_dir)
        self.evaluate(
            tf.compat.v1.variables_initializer(loaded_model.variables)
        )

        loaded = loaded_model.layers[-1]

        equal_attrs = ["name", "_expects_training_arg", "trainable"]
        for attr in equal_attrs:
            self.assertEqual(getattr(layer, attr), getattr(loaded, attr))

        all_close = ["weights", "trainable_weights", "non_trainable_weights"]
        for attr in all_close:
            self.assertAllClose(
                self.evaluate(getattr(layer, attr)),
                self.evaluate(getattr(loaded, attr)),
            )

    @test_combinations.run_with_all_model_types
    def test_trainable_layers(self):
        """Tests that trainable status of individual layers is preserved."""
        model = model = self._get_model()
        # Set the last layer to *not* be trainable.
        model.layers[-1].trainable = False
        self._train_model(model, use_dataset=True)
        loaded = self._save_and_load(model)

        self._test_evaluation(model, loaded)
        self.assertFalse(model.layers[-1].trainable)
        self.assertFalse(loaded.layers[-1].trainable)

    def test_trainable_custom_model_false(self):
        """Tests that overall False trainable status of Model is preserved."""
        # Set all layers to *not* be trainable.
        model = test_utils.SmallSubclassMLP(1, 4, trainable=False)
        model.compile(loss="mse", optimizer="rmsprop")
        self._train_model(model, use_dataset=False)
        loaded = self._save_and_load(model)

        self._test_evaluation(model, loaded)
        self.assertEmpty(model.trainable_variables)
        self.assertEmpty(loaded.trainable_variables)

    def test_maintains_losses(self):
        """Tests that the layer losses do not change before and after export."""
        model = keras.models.Sequential([LayerWithLoss()])
        model.compile(loss="mse", optimizer="rmsprop")
        input_arr = np.random.random((1, 3))
        target_arr = np.random.random((1, 3))

        # Test that symbolic losses are maintained (train_on_batch saves
        # symbolic losses.)
        model.train_on_batch(input_arr, target_arr)
        previous_losses = model.losses[:]

        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        with previous_losses[0].graph.as_default():
            # If we try to compare symbolic Tensors in eager mode assertAllEqual
            # will return False even if they are the same Tensor.
            self.assertEqual(previous_losses, model.losses)

        if tf.executing_eagerly():
            # Test that eager losses are maintained.
            model(input_arr)  # Calls model eagerly, creating eager losses.
            previous_losses = model.losses[:]
            model.save(saved_model_dir, save_format="tf")
            self.assertAllEqual(previous_losses, model.losses)

    def test_layer_with_learning_phase(self):
        layer = LayerWithLearningPhase()
        layer.build([None, None])
        saved_model_dir = self._save_model_dir()
        model = test_utils.get_model_from_layers(
            [layer], input_shape=[None], model_type="functional"
        )
        model.save(saved_model_dir, save_format="tf")
        loaded_model = keras_load.load(saved_model_dir)
        loaded = loaded_model.layers[-1]
        input_arr = tf.ones((4, 3))

        # Run the layer, and use the keras backend learning phase
        keras.backend.set_learning_phase(0)
        self.assertAllEqual(input_arr, loaded(input_arr))
        keras.backend.set_learning_phase(1)
        self.assertAllEqual(tf.zeros((4, 3)), loaded(input_arr))

        # Run the layer while explicitly setting the training argument
        self.assertAllEqual(
            input_arr, loaded(input_arr, training=tf.constant(False))
        )
        self.assertAllEqual(
            tf.zeros((4, 3)), loaded(input_arr, training=tf.constant(True))
        )

    @test_combinations.run_with_all_model_types
    def test_standard_loader(self):
        model = test_utils.get_small_mlp(1, 4, input_dim=3)
        model.activity_regularizer = regularizers.get("l2")

        def eager_loss():
            return tf.reduce_sum(model.weights[0])

        model.add_loss(eager_loss)

        # Call predict to ensure that all layers are built and inputs are set.
        model.predict(np.random.random((1, 3)).astype(np.float32))
        saved_model_dir = self._save_model_dir()

        model.save(saved_model_dir, save_format="tf")

        loaded = tf.saved_model.load(saved_model_dir)
        self.evaluate(tf.compat.v1.variables_initializer(loaded.variables))
        all_close = [
            "variables",
            "trainable_variables",
            "non_trainable_variables",
        ]
        for attr in all_close:
            self.assertAllClose(
                self.evaluate(getattr(model, attr)),
                self.evaluate(getattr(loaded.keras_api, attr)),
            )
        self.assertLen(loaded.regularization_losses, 1)
        expected_layers = len(model.layers)
        self.assertEqual(expected_layers, len(loaded.keras_api.layers))
        input_arr = tf.ones((4, 3))
        self.assertAllClose(
            self.evaluate(model(input_arr)),
            self.evaluate(loaded(input_arr, training=False)),
        )

    @test_combinations.run_with_all_model_types
    def test_compiled_model(self):
        # TODO(b/134519980): Issue with model.fit if the model call function
        # uses a tf.function (Graph mode only).
        if not tf.executing_eagerly():
            return

        input_arr = np.random.random((1, 3))
        target_arr = np.random.random((1, 4))

        model = test_utils.get_small_mlp(1, 4, input_dim=3)
        expected_predict = model.predict(input_arr)

        # Compile and save model.
        model.compile("rmsprop", "mse")
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        loaded = keras_load.load(saved_model_dir)
        actual_predict = loaded.predict(input_arr)
        self.assertAllClose(expected_predict, actual_predict)

        loss_before = loaded.evaluate(input_arr, target_arr)
        loaded.fit(input_arr, target_arr)
        loss_after = loaded.evaluate(input_arr, target_arr)
        self.assertLess(loss_after, loss_before)
        predict = loaded.predict(input_arr)

        ckpt_path = os.path.join(self.get_temp_dir(), "weights")
        loaded.save_weights(ckpt_path)

        # Ensure that the checkpoint is compatible with the original model.
        model.load_weights(ckpt_path)
        self.assertAllClose(predict, model.predict(input_arr))

    def test_metadata_input_spec(self):
        class LayerWithNestedSpec(keras.layers.Layer):
            def __init__(self):
                super().__init__()
                self.input_spec = {
                    "a": keras.layers.InputSpec(max_ndim=3, axes={-1: 2}),
                    "b": keras.layers.InputSpec(
                        shape=(None, 2, 3), dtype="int32"
                    ),
                }

            @property
            def _use_input_spec_as_call_signature(self):
                return True

        layer = LayerWithNestedSpec()
        saved_model_dir = self._save_model_dir()
        model = test_utils.get_model_from_layers([layer], model_type="subclass")
        model(
            {
                "a": tf.constant([[2, 4]]),
                "b": tf.ones([1, 2, 3], dtype=tf.int32),
            }
        )
        model.save(saved_model_dir, save_format="tf")
        loaded_model = keras_load.load(saved_model_dir)
        loaded = loaded_model.layers[-1]
        self.assertEqual(3, loaded.input_spec["a"].max_ndim)
        self.assertEqual({-1: 2}, loaded.input_spec["a"].axes)
        self.assertAllEqual([None, 2, 3], loaded.input_spec["b"].shape)
        self.assertEqual("int32", loaded.input_spec["b"].dtype)

    def test_must_restore_from_config_fails_if_layer_is_not_in_scope(self):
        class LayerThatShouldFailIfNotAdded(keras.layers.Layer):
            _must_restore_from_config = True

        layer = LayerThatShouldFailIfNotAdded()
        saved_model_dir = self._save_model_dir()
        model = test_utils.get_model_from_layers(
            [layer], input_shape=[3], model_type="functional"
        )
        model.save(saved_model_dir, save_format="tf")
        with self.assertRaisesRegex(
            ValueError, "Unknown layer: 'LayerThatShouldFailIfNotAdded'"
        ):
            _ = keras_load.load(saved_model_dir)

    def test_must_restore_from_config_custom_object_scope(self):
        class LayerThatShouldFailIfNotAdded(keras.layers.Layer):
            _must_restore_from_config = True

        layer = LayerThatShouldFailIfNotAdded()
        model = test_utils.get_model_from_layers(
            [layer], input_shape=[3], model_type="functional"
        )
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        with object_registration.CustomObjectScope(
            {"LayerThatShouldFailIfNotAdded": LayerThatShouldFailIfNotAdded}
        ):
            _ = keras_load.load(saved_model_dir)

    def test_must_restore_from_config_registration(self):
        layer = GlobalLayerThatShouldFailIfNotAdded()
        saved_model_dir = self._save_model_dir()
        model = test_utils.get_model_from_layers(
            [layer], input_shape=[3], model_type="functional"
        )
        model.save(saved_model_dir, save_format="tf")
        _ = keras_load.load(saved_model_dir)

    def test_multi_input_model(self):
        input_1 = keras.layers.Input(shape=(3,))
        input_2 = keras.layers.Input(shape=(5,))
        model = keras.Model([input_1, input_2], [input_1, input_2])
        saved_model_dir = self._save_model_dir()

        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)
        input_arr_1 = np.random.random((1, 3)).astype("float32")
        input_arr_2 = np.random.random((1, 5)).astype("float32")

        outputs = loaded([input_arr_1, input_arr_2])
        self.assertAllEqual(input_arr_1, outputs[0])
        self.assertAllEqual(input_arr_2, outputs[1])

    def test_revived_sequential(self):
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(
                5, input_shape=(3,), kernel_regularizer=regularizers.get("l2")
            )
        )
        model.add(
            keras.layers.Dense(2, kernel_regularizer=regularizers.get("l2"))
        )

        self.evaluate(tf.compat.v1.variables_initializer(model.variables))

        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)

        self.assertLen(loaded.layers, 2)
        self.assertLen(loaded.losses, 2)

        loaded.pop()

        self.assertLen(loaded.layers, 1)
        self.assertLen(loaded.losses, 1)

        loaded.add(
            keras.layers.Dense(2, kernel_regularizer=regularizers.get("l2"))
        )

        self.assertLen(loaded.layers, 2)
        self.assertLen(loaded.losses, 2)

    def testBatchNormUpdates(self):
        model = keras.models.Sequential(
            keras.layers.BatchNormalization(input_shape=(1,))
        )
        self.evaluate(tf.compat.v1.variables_initializer(model.variables))
        saved_model_dir = self._save_model_dir()

        with self.captureWritesToStream(sys.stderr) as captured_logs:
            model.save(saved_model_dir, save_format="tf")
            loaded = keras_load.load(saved_model_dir)

        # Assert that saving does not log deprecation warnings
        # (even if it needs to set learning phase for compat reasons)
        if tf.executing_eagerly():
            self.assertNotIn("deprecated", captured_logs.contents())

        input_arr = tf.constant([[11], [12], [13]], dtype=tf.float32)
        input_arr2 = tf.constant([[14], [15], [16]], dtype=tf.float32)
        self.assertAllClose(self.evaluate(loaded.layers[-1].moving_mean), [0])

        self.evaluate(loaded(input_arr, training=True))
        if not tf.executing_eagerly():
            self.evaluate(loaded.get_updates_for(input_arr))
        self.assertAllClose(
            self.evaluate(loaded.layers[-1].moving_mean), [0.12]
        )

        self.evaluate(loaded(input_arr2, training=False))
        if not tf.executing_eagerly():
            self.evaluate(loaded.get_updates_for(input_arr2))
        self.assertAllClose(
            self.evaluate(loaded.layers[-1].moving_mean), [0.12]
        )

    def testDisablingBatchNormTrainableBeforeSaving(self):
        # We disable trainable on the batchnorm layers before saving
        model = keras.models.Sequential(
            keras.layers.BatchNormalization(input_shape=(1,))
        )
        model.trainable = False
        self.evaluate(tf.compat.v1.variables_initializer(model.variables))
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)
        self.evaluate(tf.compat.v1.variables_initializer(loaded.variables))
        input_arr = tf.constant([[11], [12], [13]], dtype=tf.float32)
        input_arr2 = tf.constant([[14], [15], [16]], dtype=tf.float32)
        self.assertAllClose(self.evaluate(loaded.layers[-1].moving_mean), [0])

        # Trainable should still be disabled after loading
        self.evaluate(loaded(input_arr, training=True))
        if not tf.executing_eagerly():
            self.evaluate(loaded.get_updates_for(input_arr))
        self.assertAllClose(self.evaluate(loaded.layers[-1].moving_mean), [0.0])

        # Re-enabling trainable on the loaded model should cause the batchnorm
        # layer to start training again.
        # Note: this only works in v2.
        if tf.executing_eagerly():
            loaded.trainable = True
            self.evaluate(loaded(input_arr, training=True))
            self.assertAllClose(
                self.evaluate(loaded.layers[-1].moving_mean), [0.12]
            )

            self.evaluate(loaded(input_arr2, training=False))
            self.assertAllClose(
                self.evaluate(loaded.layers[-1].moving_mean), [0.12]
            )

    def testSaveWithSignatures(self):
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(
                5, input_shape=(3,), kernel_regularizer=regularizers.get("l2")
            )
        )
        model.add(keras.layers.Dropout(0.5))
        model.add(
            keras.layers.Dense(4, kernel_regularizer=regularizers.get("l2"))
        )

        input_arr = np.random.random((2, 3))
        target_arr = np.random.random((2, 4))

        model.compile(loss="mse", optimizer="rmsprop")
        model.train_on_batch(input_arr, target_arr)

        @tf.function(input_signature=[tf.TensorSpec((None, 3))])
        def predict(inputs):
            return {"predictions": model(inputs)}

        feature_configs = {
            "inputs": tf.io.FixedLenFeature(shape=[2, 3], dtype=tf.float32)
        }

        @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
        def parse_and_predict(examples):
            features = tf.compat.v1.parse_single_example(
                examples[0], feature_configs
            )
            return {
                "predictions": model(features["inputs"]),
                "layer_1_outputs": model.layers[0](features["inputs"]),
            }

        saved_model_dir = self._save_model_dir()
        model.save(
            saved_model_dir,
            save_format="tf",
            signatures={
                "predict": predict,
                "parse_and_predict": parse_and_predict,
            },
        )
        model.save(
            "/tmp/saved",
            save_format="tf",
            signatures={
                "predict": predict,
                "parse_and_predict": parse_and_predict,
            },
        )

        loaded = keras_load.load(saved_model_dir)

        self.assertAllClose(
            model.predict(input_arr),
            loaded.signatures["predict"](
                tf.convert_to_tensor(input_arr.astype("float32"))
            )["predictions"],
        )

        feature = {
            "inputs": feature_pb2.Feature(
                float_list=feature_pb2.FloatList(
                    value=input_arr.astype("float32").flatten()
                )
            )
        }
        example = example_pb2.Example(
            features=feature_pb2.Features(feature=feature)
        )
        outputs = loaded.signatures["parse_and_predict"](
            tf.convert_to_tensor([example.SerializeToString()])
        )
        self.assertAllClose(model.predict(input_arr), outputs["predictions"])
        self.assertAllClose(
            model.layers[0](input_arr), outputs["layer_1_outputs"]
        )

    def testTrainingDefaults(self):
        def assert_training_default(fn, default_value):
            arg_spec = tf_inspect.getfullargspec(fn)
            fn_defaults = arg_spec.defaults or []
            defaults = dict()
            # The call arg defaults are an n-tuple of the last n elements of the
            # args list. (n = # of elements that have a default argument)
            for i in range(-1 * len(fn_defaults), 0):
                defaults[arg_spec.args[i]] = fn_defaults[i]
            # The default training arg will be any (non-None) default specified
            # in the method signature, or None if no value is specified.
            defaults.update(arg_spec.kwonlydefaults or {})
            self.assertEqual(defaults["training"], default_value)

        class LayerWithTrainingRequiredArg(keras.engine.base_layer.Layer):
            def call(self, inputs, training):
                return control_flow_util.smart_cond(
                    training, lambda: inputs * 0, lambda: tf.identity(inputs)
                )

        class LayerWithTrainingDefaultTrue(keras.engine.base_layer.Layer):
            def call(self, inputs, training=True):
                return control_flow_util.smart_cond(
                    training, lambda: inputs * 0, lambda: tf.identity(inputs)
                )

        class Model(keras.models.Model):
            def __init__(self):
                super().__init__()
                self.layer_with_training_default_none = LayerWithLearningPhase()
                self.layer_with_training_default_true = (
                    LayerWithTrainingDefaultTrue()
                )
                self.layer_with_required_training_arg = (
                    LayerWithTrainingRequiredArg()
                )

            def call(self, inputs):
                x = self.layer_with_training_default_none(inputs)
                x += self.layer_with_training_default_true(inputs)
                x += self.layer_with_required_training_arg(inputs, False)
                return x

        model = Model()
        # Build and set model inputs
        model.predict(np.ones([1, 3]).astype("float32"))
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        load = tf.saved_model.load(saved_model_dir)

        # Ensure that the Keras loader is able to load and build the model.
        _ = keras_load.load(saved_model_dir)

        assert_training_default(load.__call__, False)
        assert_training_default(
            load.layer_with_training_default_none.__call__, False
        )
        assert_training_default(
            load.layer_with_training_default_true.__call__, True
        )

        # Assert that there are no defaults for layer with required training arg
        arg_spec = tf_inspect.getfullargspec(
            load.layer_with_required_training_arg.__call__
        )
        self.assertFalse(arg_spec.defaults)  # defaults is None or empty

    def testTraceModelWithKwarg(self):
        class Model(keras.models.Model):
            def call(self, inputs, keyword=None):
                return tf.identity(inputs)

        model = Model()
        prediction = model.predict(np.ones([1, 3]).astype("float32"))
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        with object_registration.custom_object_scope({"Model": Model}):
            loaded = keras_load.load(saved_model_dir)
        self.assertAllClose(
            prediction, loaded.predict(np.ones([1, 3]).astype("float32"))
        )

        loaded_without_scope = keras_load.load(saved_model_dir)
        if tf.__internal__.tf2.enabled():
            with self.assertRaises(NotImplementedError):
                loaded_without_scope.predict(np.ones([1, 3]).astype("float32"))

    def testFeatureColumns(self):
        # TODO(b/120099662): Error with table initialization with Keras models
        # in graph mode.
        if tf.executing_eagerly():
            numeric = tf.feature_column.numeric_column("a")
            bucketized = tf.feature_column.bucketized_column(
                numeric, boundaries=[5, 10, 15]
            )
            cat_vocab = (
                tf.feature_column.categorical_column_with_vocabulary_list(
                    "b", ["1", "2", "3"]
                )
            )
            one_hot = tf.feature_column.indicator_column(cat_vocab)
            embedding = tf.feature_column.embedding_column(
                cat_vocab, dimension=8
            )
            feature_layer = DenseFeatures([bucketized, one_hot, embedding])
            model = keras.models.Sequential(feature_layer)

            features = {"a": np.array([13, 15]), "b": np.array(["1", "2"])}
            predictions = model.predict(features)

            saved_model_dir = self._save_model_dir()
            model.save(saved_model_dir, save_format="tf")
            loaded = keras_load.load(saved_model_dir)
            loaded_predictions = loaded.predict(features)
            self.assertAllClose(predictions, loaded_predictions)

    def testSaveTensorKwarg(self):
        class LayerWithTensorKwarg(keras.layers.Layer):
            def call(self, inputs, tensor=None):
                if tensor is not None:
                    return inputs * tf.cast(tensor, tf.float32)
                else:
                    return inputs

        t = self.evaluate(tf.sequence_mask(1))
        inputs = keras.layers.Input(shape=(3))
        model = keras.models.Model(inputs, LayerWithTensorKwarg()(inputs, t))

        input_arr = np.random.random((1, 3))
        predictions = model.predict(input_arr)

        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)
        loaded_predictions = loaded.predict(input_arr)
        self.assertAllClose(predictions, loaded_predictions)

    def testModelWithTfFunctionCall(self):
        class Subclass(keras.models.Model):
            @tf.function
            def call(self, inputs, training=False):
                return inputs * tf.cast(training, tf.float32)

        model = Subclass()
        model.predict(tf.ones((1, 2)), steps=1)
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)
        self.assertAllEqual(
            [[1, 5]],
            self.evaluate(loaded(tf.constant([[1, 5.0]]), training=True)),
        )
        self.assertAllEqual(
            [[0, 0]],
            self.evaluate(loaded(tf.constant([[1, 5.0]]), training=False)),
        )

    def testReviveFunctionalModel(self):
        class CustomAdd(keras.layers.Add):
            def build(self, input_shape):
                self.w = self.add_weight("w", shape=[])
                super().build(input_shape)

            def call(self, inputs):
                outputs = super().call(inputs)
                return outputs * self.w

        input1 = keras.layers.Input(shape=(None, 3), name="input_1")
        input2 = keras.layers.Input(shape=(None, 3), name="input_2")

        d = keras.layers.Dense(4, name="dense_with_two_inbound_nodes")
        output1 = d(input1)
        output2 = d(input2)

        # Use a custom layer in this model to ensure that layers aren't being
        # recreated directly from the config.
        outputs = CustomAdd(name="custom")([output1, output2])
        model = keras.models.Model([input1, input2], outputs, name="save_model")

        self.evaluate(tf.compat.v1.variables_initializer(model.variables))
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        loaded = keras_load.load(saved_model_dir)
        self.assertEqual("save_model", loaded.name)
        self.assertLen(
            loaded.get_layer("dense_with_two_inbound_nodes")._inbound_nodes, 2
        )
        self.assertEqual("CustomAdd", type(loaded.get_layer("custom")).__name__)
        self.assertLen(loaded.get_layer("custom").weights, 1)

    def _testAddUpdate(self, scope):
        with scope:
            layer_with_update = LayerWithUpdate()
            model = test_utils.get_model_from_layers(
                [layer_with_update], input_shape=(3,)
            )

            x = np.ones((10, 3))
            if test_utils.get_model_type() == "subclass":
                model.predict(x, batch_size=10)
            self.evaluate(tf.compat.v1.variables_initializer(model.variables))
            saved_model_dir = self._save_model_dir()
            model.save(saved_model_dir, save_format="tf")

        loaded = keras_load.load(saved_model_dir)
        loaded_layer = loaded.layers[-1]
        self.evaluate(tf.compat.v1.variables_initializer(loaded.variables))
        self.assertEqual(self.evaluate(loaded_layer.v), 0.0)

        loaded.compile("sgd", "mse")
        loaded.fit(x, x, batch_size=10)
        self.assertEqual(self.evaluate(loaded_layer.v), 1.0)

    @test_combinations.run_with_all_model_types
    def testSaveLayerWithUpdates(self):
        @tf_contextlib.contextmanager
        def nullcontextmanager():
            yield

        self._testAddUpdate(nullcontextmanager())

    @test_combinations.run_with_all_model_types
    def testSaveInStrategyScope(self):
        self._testAddUpdate(tf.distribute.MirroredStrategy().scope())

    def testSaveTimeDistributedLayer(self):
        model = keras.Sequential(
            [
                keras.layers.TimeDistributed(
                    keras.layers.Dense(
                        1, kernel_regularizer=regularizers.get("l2")
                    ),
                    input_shape=(None, 1),
                )
            ]
        )
        predictions = model.predict_on_batch(tf.ones((3, 2, 1)))

        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        loaded = keras_load.load(saved_model_dir)
        self.assertAllClose(
            loaded.predict_on_batch(tf.ones((3, 2, 1))), predictions
        )

    @parameterized.named_parameters(
        [("with_unrolling", True), ("no_unrolling", False)]
    )
    def testSaveStatefulRNN(self, unroll):
        batch = 12
        timesteps = 10
        input_dim = 8
        input_arr = np.ones((batch, timesteps, input_dim)).astype("float32")

        cells = [keras.layers.LSTMCell(32), keras.layers.LSTMCell(64)]
        if unroll:
            x = keras.Input(batch_shape=(batch, timesteps, input_dim))
        else:
            x = keras.Input(batch_shape=(batch, None, input_dim))
        layer = keras.layers.RNN(cells, stateful=True, unroll=unroll)
        y = layer(x)

        model = keras.Model(x, y)
        model.compile(
            "rmsprop", "mse", run_eagerly=test_utils.should_run_eagerly()
        )
        model.train_on_batch(
            np.zeros((batch, timesteps, input_dim)).astype("float32"),
            np.zeros((batch, 64)).astype("float32"),
        )

        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        loaded = keras_load.load(saved_model_dir)
        loaded_layer = loaded.layers[1]

        if not tf.executing_eagerly():
            keras.backend.get_session()  # force variable initialization

        self.assertAllClose(layer.states, loaded_layer.states)
        self.assertAllClose(model(input_arr), loaded(input_arr))

    def testSaveBidirectionalLSTM(self):
        # Make sure that the input spec of an unrolled RNN is not used when
        # wrapped in a Bidirectional layer.
        # https://github.com/keras-team/keras/issues/15454
        input_layer = keras.Input(
            batch_input_shape=(1, 15, 128), name="input", dtype=tf.float32
        )
        lstm_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=64,
                name="lstm",
                dropout=0.2,
                trainable=False,
                unroll=True,
            )
        )
        output_layer = lstm_layer(input_layer)
        model = keras.Model(input_layer, output_layer)
        saved_model_dir = self._save_model_dir()
        self.evaluate(tf.compat.v1.variables_initializer(model.variables))
        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)
        input_arr = np.random.random((1, 15, 128)).astype("float32")
        self.assertAllClose(model(input_arr), loaded(input_arr))

    @parameterized.named_parameters([("stateful", True), ("stateless", False)])
    def testSaveConvLSTM2D(self, stateful):
        data_format = "channels_first"
        batch, timesteps, channels, rows, cols = 12, 10, 8, 4, 4
        input_arr = np.ones((batch, timesteps, channels, rows, cols)).astype(
            "float32"
        )
        layer = keras.layers.ConvLSTM2D(
            filters=16,
            kernel_size=(1, 1),
            data_format=data_format,
            stateful=stateful,
        )
        x = keras.Input(batch_shape=(batch, timesteps, channels, rows, cols))
        y = layer(x)
        model = keras.Model(x, y)

        predict_1 = model(input_arr)
        self.evaluate([v.initializer for v in model.variables])
        saved_model_dir = self._save_model_dir()

        model.save(saved_model_dir, save_format="tf")
        del model

        loaded = keras_load.load(saved_model_dir)
        self.evaluate([v.initializer for v in loaded.variables])
        if stateful:
            loaded.reset_states()
        predict_2 = loaded(input_arr)
        self.assertAllClose(predict_1, predict_2)

    def testSaveWithRaggedInputs(self):
        class EmbeddingMerger(keras.layers.Layer):
            def __init__(self, list_features, **kwargs):
                super().__init__(**kwargs)
                self._supports_ragged_inputs = True
                self.embeddings = {
                    feature: keras.layers.Embedding(10, 3)
                    for feature in list_features
                }
                self.mean = keras.layers.Lambda(
                    tf.reduce_mean, arguments=dict(axis=1)
                )

            def call(self, inputs):
                tensors = [self.embeddings[col](inputs[col]) for col in inputs]
                tensors = [self.mean(inp) for inp in tensors]
                return keras.layers.Add()(tensors)

        list_features = ["feature_1", "feature_2"]
        feature_1 = tf.ragged.constant([[0.0], [1, 3]])
        feature_2 = tf.ragged.constant([[1.0, 2], [4]])
        f = {"feature_1": feature_1, "feature_2": feature_2}
        f_inputs = {
            "feature_1": keras.Input(
                shape=(None,), name="feature_1", ragged=True
            ),
            "feature_2": keras.Input(
                shape=(None,), name="feature_2", ragged=True
            ),
        }

        out = EmbeddingMerger(list_features)(f_inputs)
        model = keras.Model(f_inputs, out)
        self.evaluate(tf.compat.v1.variables_initializer(model.variables))
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        loaded = keras_load.load(saved_model_dir)
        self.evaluate(tf.compat.v1.variables_initializer(loaded.variables))
        self.assertAllClose(model.predict(f), loaded.predict(f))

    def testSaveMultipleInputs(self):
        class CustomLayer(keras.layers.Layer):
            def call(self, *input_list):
                self.add_loss(input_list[-2] * 2)
                return sum(
                    input_list[:-1]
                )  # The test's last input is a non-tensor arg

        class CustomModel(keras.Model):
            def build(self, _):
                self.layer = CustomLayer()

            def call(self, *inputs):
                inputs = list(inputs)
                inputs.append(
                    object()
                )  # Test that the layer handles non-tensor inputs
                return self.layer(*inputs)

        model = CustomModel()
        inp = [
            tf.constant(i, shape=[1, 1], dtype=tf.float32) for i in range(1, 5)
        ]
        expected = model(*inp)
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)
        actual = loaded(*inp)
        self.assertAllEqual(self.evaluate(expected), self.evaluate(actual))

    def testSaveMultipleInputsWithTraining(self):
        class CustomModel(keras.Model):
            def call(self, input_1, training, input_2):
                if training:
                    return input_1
                else:
                    return input_2

        inp1 = tf.constant(1.0, shape=[1])
        inp2 = tf.constant(2.0, shape=[1])

        model = CustomModel()
        self.assertEqual(self.evaluate(model(inp1, True, inp2)), 1.0)
        self.assertEqual(self.evaluate(model(inp1, False, inp2)), 2.0)

        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)
        self.assertEqual(self.evaluate(loaded(inp1, True, inp2)), 1.0)
        self.assertEqual(self.evaluate(loaded(inp1, False, inp2)), 2.0)

    def test_wrapped_layer_training(self):
        class Custom(keras.models.Model):
            def __init__(self):
                super().__init__()
                self.layer = LayerWithLearningPhase()

            def call(self, inputs):
                return self.layer(inputs)

        model = Custom()
        x = tf.constant(1.0, shape=[1, 1])
        expected_default = model(x)
        expected_training_true = model(x, training=True)
        expected_training_false = model(x, training=False)
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        def assert_loaded_model(loaded):
            actual_default = loaded(x)
            actual_training_true = loaded(x, training=True)
            actual_training_false = loaded(x, training=False)
            self.assertAllClose(
                [
                    expected_default,
                    expected_training_true,
                    expected_training_false,
                ],
                [actual_default, actual_training_true, actual_training_false],
            )

        assert_loaded_model(keras_load.load(saved_model_dir))
        assert_loaded_model(tf.saved_model.load(saved_model_dir))

    @parameterized.named_parameters([("true", True), ("false", False)])
    def test_save_layer_autocast(self, autocast):
        class CustomLayer(keras.layers.Layer):
            def __init__(self):
                super().__init__(autocast=autocast)

        class CustomModel(keras.Model):
            def __init__(self):
                super().__init__(autocast=autocast)

            def call(self, inputs):
                return inputs

        x = tf.constant([3], dtype=tf.float64)

        x_in = keras.Input((1,))
        output = CustomLayer()(x_in)
        output = CustomModel()(output)
        model = keras.Model(inputs=x_in, outputs=output)

        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")
        loaded = keras_load.load(saved_model_dir)
        self.assertEqual(autocast, loaded.layers[-1]._autocast)
        self.assertEqual(autocast, loaded.layers[-2]._autocast)
        self.assertEqual(self.evaluate(model(x)), self.evaluate(loaded(x)))


class TestSavedModelFormat(tf.test.TestCase):
    def _save_model_dir(self, dirname="saved_model"):
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        return os.path.join(temp_dir, dirname)

    def test_load_with_custom_model_and_layer(self):
        class CustomLayer(keras.layers.Layer):
            def __call__(self, inputs):
                return inputs

        class Model(keras.models.Model):
            def __init__(self):
                super().__init__()
                self.layer = CustomLayer()  # noqa: F821

            @tf.function(input_signature=[tf.TensorSpec([None, 1])])
            def call(self, inputs):
                return self.layer(inputs)

        model = Model()
        inp = tf.constant([[1.0]])
        model(inp)
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        # Even if the `CustomLayer` is not provided in `custom_object_scope`,
        # `Model` still has that reference.
        with object_registration.custom_object_scope({"Model": Model}):
            loaded = keras_load.load(saved_model_dir)
        self.assertAllEqual([[1.0]], self.evaluate(loaded(inp)))
        self.assertAllEqual([[1.0]], self.evaluate(loaded.layer(inp)))
        self.assertIsInstance(loaded.layer, CustomLayer)

        # If `CustomLayer` is provided in `custom_object_scope`, it should of
        # course use that custom class.
        with object_registration.custom_object_scope(
            {"Model": Model, "CustomLayer": CustomLayer}
        ):
            loaded = keras_load.load(saved_model_dir)
        self.assertAllEqual([[1.0]], self.evaluate(loaded(inp)))
        self.assertAllEqual([[1.0]], self.evaluate(loaded.layer(inp)))
        self.assertIsInstance(loaded.layer, CustomLayer)

    def test_save_without_tracing(self):
        class DoNotTrace(keras.layers.Layer):
            def __init__(self):
                super().__init__()
                self.input_spec = keras.layers.InputSpec(shape=[None])
                self.built = True

            def call(self, inputs):
                raise ValueError("I said do not trace")

            def get_config(self):
                return {}

            @property
            def _use_input_spec_as_call_signature(self):
                return True

        root = keras.models.Sequential()
        root.add(keras.layers.Input(shape=(3,)))
        root.attached_layer = DoNotTrace()

        saved_model_dir = self._save_model_dir()

        # With the default settings, the call function is traced.
        with self.assertRaisesRegex(ValueError, "do not trace"):
            root.save(saved_model_dir, save_format="tf")

        # When saving the config only, the layer call function should not be not
        # traced.
        root.save(saved_model_dir, save_format="tf", save_traces=False)
        loaded = tf.saved_model.load(saved_model_dir)
        self.assertTrue(hasattr(loaded, "attached_layer"))

        # This should raise an error when loaded without the custom object
        loaded = keras_load.load(saved_model_dir)
        with self.assertRaisesRegex(ValueError, "Cannot call custom layer"):
            loaded.attached_layer(tf.constant([1.0]))

        # Try loading with the custom objects
        with object_registration.CustomObjectScope({"DoNotTrace": DoNotTrace}):
            loaded = keras_load.load(saved_model_dir)
        with self.assertRaisesRegex(ValueError, "I said do not trace"):
            loaded.attached_layer(tf.constant([1.0]))

    def test_load_non_keras_saved_model(self):
        model = test_utils.get_small_functional_mlp(1, 4, input_dim=3)
        saved_model_dir = self._save_model_dir()
        tf.saved_model.save(model, saved_model_dir)
        with self.assertRaisesRegex(
            ValueError, "Unable to create a Keras model"
        ):
            keras_load.load(saved_model_dir)

    def test_random_generator_custom_layer(self):
        class CustomDropout(keras.layers.Layer):
            def __init__(self, dropout_rate=0.1, **kwargs):
                super().__init__(**kwargs)
                self.dropout_rate = dropout_rate
                self.dropout = keras.layers.Dropout(
                    dropout_rate, rng_type="stateful"
                )

            def call(self, inputs, training=False):
                return self.dropout(inputs, training=training)

        root = keras.models.Sequential(
            [keras.layers.Input(shape=(3,)), CustomDropout()]
        )
        saved_model_dir = self._save_model_dir()
        root.save(saved_model_dir, save_format="tf")

        loaded = keras_load.load(saved_model_dir)

        output = loaded(tf.random.uniform([1, 3]), training=True)
        self.assertAllEqual([1, 3], output.shape)

    def test_random_generator_with_tracing(self):
        # This test is to ensure we trace the training = True function first,
        # otherwise tf.function will raise error about creating variables in the
        # non-first call.
        class LayerWithDropout(keras.layers.Layer):
            def __init__(self, dropout_rate):
                super().__init__()
                self.dropout_rate = dropout_rate
                self.dropout_layer = keras.layers.Dropout(self.dropout_rate)

            def call(self, inputs, training=None):
                if not training:
                    return inputs
                else:
                    return self.dropout_layer(inputs, training=training)

        root = keras.models.Sequential(
            [keras.layers.Input(shape=(3,)), LayerWithDropout(0.1)]
        )
        saved_model_dir = self._save_model_dir()
        root.save(saved_model_dir, save_format="tf")

        loaded = keras_load.load(saved_model_dir)

        output = loaded(tf.random.uniform([1, 3]), training=True)
        self.assertAllEqual([1, 3], output.shape)


class TestLayerCallTracing(tf.test.TestCase, parameterized.TestCase):
    def test_functions_have_same_trace(self):
        class Layer(keras.engine.base_layer.Layer):
            def call(self, inputs):
                return inputs

            def call2(self, inputs):
                return inputs * 2

        layer = Layer()

        call_collection = keras_save.LayerCallCollection(layer)
        fn = call_collection.add_function(layer.call, "call", True)
        fn2 = call_collection.add_function(layer.call2, "call2", True)

        with keras_save.tracing_scope():
            fn(np.ones((2, 3)))
            fn(np.ones((4, 5)))

        self.assertLen(
            fn.wrapped_call._list_all_concrete_functions_for_serialization(), 2
        )
        self.assertLen(
            fn2.wrapped_call._list_all_concrete_functions_for_serialization(), 2
        )

        # Check that the shapes are correct
        self.assertEqual(
            {(2, 3), (4, 5)},
            set(
                tuple(c.structured_input_signature[0][0].shape.as_list())
                for c in fn2.wrapped_call._list_all_concrete_functions_for_serialization()  # noqa: E501
            ),
        )

    def test_training_arg_replacement(self):
        def assert_num_traces(layer_cls, training_keyword):
            layer = layer_cls()
            call_collection = keras_save.LayerCallCollection(layer)
            fn = call_collection.add_function(layer.call, "call", True)

            with keras_save.tracing_scope():
                fn(np.ones((2, 3)), training=True)
            self.assertLen(
                fn.wrapped_call._list_all_concrete_functions_for_serialization(),  # noqa: E501
                2,
            )
            with keras_save.tracing_scope():
                fn(np.ones((2, 4)), training=False)
            self.assertLen(
                fn.wrapped_call._list_all_concrete_functions_for_serialization(),  # noqa: E501
                4,
            )

            if training_keyword:
                with keras_save.tracing_scope():
                    fn(np.ones((2, 5)), True)
                self.assertLen(
                    fn.wrapped_call._list_all_concrete_functions_for_serialization(),  # noqa: E501
                    6,
                )
                with keras_save.tracing_scope():
                    fn(np.ones((2, 6)))
                self.assertLen(
                    fn.wrapped_call._list_all_concrete_functions_for_serialization(),  # noqa: E501
                    8,
                )

        class LayerWithTrainingKeyword(keras.engine.base_layer.Layer):
            def call(self, inputs, training=False):
                return inputs * training

        assert_num_traces(LayerWithTrainingKeyword, training_keyword=True)

        class LayerWithKwargs(keras.engine.base_layer.Layer):
            def call(self, inputs, **kwargs):
                return inputs * kwargs["training"]

        assert_num_traces(LayerWithKwargs, training_keyword=False)

        class LayerWithChildLayer(keras.engine.base_layer.Layer):
            def __init__(self):
                self.child = LayerWithKwargs()
                super().__init__()

            def call(self, inputs):
                return self.child(inputs)

        assert_num_traces(LayerWithChildLayer, training_keyword=False)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_maintains_losses(self):
        layer = LayerWithLoss()
        layer(np.ones((2, 3)))
        previous_losses = layer.losses[:]

        call_collection = keras_save.LayerCallCollection(layer)
        fn = call_collection.add_function(layer.call, "call", True)
        fn(np.ones((2, 3)))

        self.assertAllEqual(
            self.evaluate(previous_losses), self.evaluate(layer.losses)
        )


@object_registration.register_keras_serializable("Testing")
class CustomMeanMetric(keras.metrics.Mean):
    def update_state(self, *args):
        # Sometimes built-in metrics return an op in update_state. Custom
        # metrics don't support returning ops, so wrap the update_state method
        # while returning nothing.
        super().update_state(*args)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class MetricTest(tf.test.TestCase, parameterized.TestCase):
    def _save_model_dir(self, dirname="saved_model"):
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        return os.path.join(temp_dir, dirname)

    def generate_inputs(self, num_tensor_args, shape=(1, 5)):
        return [
            np.random.uniform(0, 1, shape).astype("float32")
            for _ in range(num_tensor_args)
        ]

    def _test_metric_save_and_load(
        self,
        metric,
        save_dir,
        num_tensor_args,
        shape=(1, 5),
        test_sample_weight=True,
    ):
        with self.cached_session():
            model = test_utils.get_model_from_layers(
                [keras.layers.Layer()], input_shape=[3], model_type="functional"
            )
            model.saved_metric = metric
            model.save(save_dir, save_format="tf")
            loaded_model = keras_load.load(save_dir)
            loaded = loaded_model.saved_metric
            self.evaluate([v.initializer for v in loaded.variables])
            self.assertEqual(metric.name, loaded.name)
            self.assertEqual(metric.dtype, loaded.dtype)

            inputs = self.generate_inputs(num_tensor_args, shape)
            actual = self.evaluate(metric(*inputs))
            self.assertAllClose(actual, loaded(*inputs))
            self.assertAllClose(metric.variables, loaded.variables)

            # Test with separate calls to update state and result.
            inputs = self.generate_inputs(num_tensor_args, shape)
            self.evaluate(metric.update_state(*inputs))
            self.evaluate(loaded.update_state(*inputs))
            actual = self.evaluate(metric.result())
            self.assertAllClose(actual, loaded.result())

            if test_sample_weight:
                # Test with sample weights input.
                inputs = self.generate_inputs(num_tensor_args, shape)
                sample_weight = self.generate_inputs(1, [])[0]
                inputs.append(sample_weight)

                actual = self.evaluate(metric(*inputs))
                self.assertAllClose(actual, loaded(*inputs))
            return loaded

    @parameterized.named_parameters(
        [
            ("mean", keras.metrics.Mean, 1, (1, 5)),
            ("false_positives", keras.metrics.FalsePositives, 2, (1, 5)),
            (
                "precision_at_top_k",
                keras.metrics.Precision,
                2,
                (2, 3, 4),
                {"top_k": 2, "class_id": 1},
            ),
            (
                "precision_at_recall",
                keras.metrics.PrecisionAtRecall,
                2,
                (1, 5),
                {"recall": 0.8},
            ),
            ("auc", keras.metrics.AUC, 2, (1, 5), {"multi_label": True}),
            ("cosine_similarity", keras.metrics.CosineSimilarity, 2, (2, 3, 1)),
        ]
    )
    def test_metric(self, metric_cls, num_tensor_args, shape, init_kwargs=None):
        init_kwargs = init_kwargs or {}
        metric = metric_cls(**init_kwargs)
        metric(*self.generate_inputs(num_tensor_args, shape))
        self.evaluate([v.initializer for v in metric.variables])
        loaded = self._test_metric_save_and_load(
            metric, self._save_model_dir(), num_tensor_args, shape
        )
        self.assertEqual(type(loaded), type(metric))

    @parameterized.named_parameters(
        [
            ("mean", keras.metrics.Mean, 1, False),
            ("auc", keras.metrics.AUC, 2, False),
            ("mean_tensor", keras.metrics.MeanTensor, 1, True),
        ]
    )
    def test_custom_metric(self, base_cls, num_tensor_args, requires_build):
        class CustomMetric(base_cls):
            def update_state(self, *args):
                # Sometimes built-in metrics return an op in update_state.
                # Custom metrics don't support returning ops, so wrap the
                # update_state method while returning nothing.
                super().update_state(*args)

        with self.cached_session():
            metric = CustomMetric()
            save_dir = self._save_model_dir("first_save")

            if requires_build:
                metric(*self.generate_inputs(num_tensor_args))

            self.evaluate([v.initializer for v in metric.variables])

            with self.assertRaisesRegex(
                ValueError, "Unable to restore custom object"
            ):
                self._test_metric_save_and_load(
                    metric, save_dir, num_tensor_args
                )
            with object_registration.CustomObjectScope(
                {"CustomMetric": CustomMetric}
            ):
                loaded = self._test_metric_save_and_load(
                    metric, save_dir, num_tensor_args, test_sample_weight=False
                )

                self._test_metric_save_and_load(
                    loaded,
                    self._save_model_dir("second_save"),
                    num_tensor_args,
                    test_sample_weight=False,
                )

    def test_registered_custom_metric(self):

        with self.cached_session():
            metric = CustomMeanMetric()
            save_dir = self._save_model_dir("first_save")
            self.evaluate([v.initializer for v in metric.variables])
            loaded = self._test_metric_save_and_load(
                metric, save_dir, num_tensor_args=1, test_sample_weight=False
            )

            self._test_metric_save_and_load(
                loaded,
                self._save_model_dir("second_save"),
                num_tensor_args=1,
                test_sample_weight=False,
            )

    def test_custom_metric_wrapped_call(self):
        class NegativeMean(keras.metrics.Mean):
            @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
            def update_state(self, value):
                super().update_state(-value)

        metric = NegativeMean()
        self.evaluate([v.initializer for v in metric.variables])
        with object_registration.CustomObjectScope(
            {"NegativeMean": NegativeMean}
        ):
            self._test_metric_save_and_load(
                metric, self._save_model_dir(), 1, test_sample_weight=False
            )

    @test_combinations.run_with_all_model_types
    def test_custom_metric_model(self):
        # TODO(b/134519980): Issue with `model.fit` if the model call function
        # uses a `tf.function` in graph mode.
        if not tf.executing_eagerly():
            return

        x = np.random.random((1, 3))
        y = np.random.random((1, 4))

        class CustomMetric(keras.metrics.MeanSquaredError):
            pass

        def zero_metric(y_true, y_pred):
            del y_true, y_pred
            return 0

        model = test_utils.get_small_mlp(1, 4, input_dim=3)
        model.compile(
            loss="mse", optimizer="SGD", metrics=[CustomMetric(), zero_metric]
        )
        model.fit(x, y)
        saved_model_dir = self._save_model_dir()
        model.save(saved_model_dir, save_format="tf")

        with self.assertRaisesRegex(ValueError, "custom_objects"):
            keras_load.load(saved_model_dir)

        with object_registration.CustomObjectScope(
            {"CustomMetric": CustomMetric, "zero_metric": zero_metric}
        ):
            loaded = keras_load.load(saved_model_dir)

        self.evaluate([v.initializer for v in loaded.variables])
        loaded.fit(x, y)


class TestUpdateMetadata(tf.test.TestCase):
    def testAddFullSaveSpec(self):
        save_spec = tf.TensorSpec([3, 5], dtype=tf.int32)
        node_metadata = json_utils.Encoder().encode({"save_spec": save_spec})

        metadata = saved_metadata_pb2.SavedMetadata()
        metadata.nodes.add(
            version=versions_pb2.VersionDef(
                producer=1, min_consumer=1, bad_consumers=[]
            ),
            identifier="_tf_keras_model",
            metadata=node_metadata,
        )

        new_metadata = keras_load._update_to_current_version(metadata)
        node_metadata = json_utils.decode(new_metadata.nodes[0].metadata)
        expected_full_spec = ([tf.TensorSpec(shape=(3, 5), dtype=tf.int32)], {})
        self.assertAllEqual(
            expected_full_spec, node_metadata.get("full_save_spec")
        )


if __name__ == "__main__":
    with saved_model_utils.keras_option_scope(
        save_traces=False, in_tf_saved_model_scope=True
    ):
        tf.test.main()
