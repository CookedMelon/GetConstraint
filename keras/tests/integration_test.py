# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for Keras."""

import os
import random

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras import utils
from keras.layers.rnn import legacy_cells
from keras.legacy_tf_layers import base as base_layer
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


class KerasIntegrationTest(test_combinations.TestCase):
    def _save_and_reload_model(self, model):
        self.temp_dir = self.get_temp_dir()
        fpath = os.path.join(
            self.temp_dir, f"test_model_{random.randint(0, 10000000.0)}"
        )
        if tf.executing_eagerly():
            save_format = "tf"
        else:
            if (
                not isinstance(model, keras.Sequential)
                and not model._is_graph_network
            ):
                return model  # Not supported
            save_format = "h5"
        model.save(fpath, save_format=save_format)
        model = keras.models.load_model(fpath)
        return model


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class VectorClassificationIntegrationTest(test_combinations.TestCase):
    def test_vector_classification(self):
        np.random.seed(1337)
        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=100, test_samples=0, input_shape=(10,), num_classes=2
        )
        y_train = utils.to_categorical(y_train)

        model = test_utils.get_model_from_layers(
            [
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(y_train.shape[-1], activation="softmax"),
            ],
            input_shape=x_train.shape[1:],
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.legacy.adam.Adam(0.005),
            metrics=["acc"],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=10,
            validation_data=(x_train, y_train),
            verbose=2,
        )
        self.assertGreater(history.history["val_acc"][-1], 0.7)
        _, val_acc = model.evaluate(x_train, y_train)
        self.assertAlmostEqual(history.history["val_acc"][-1], val_acc)
        predictions = model.predict(x_train)
        self.assertEqual(predictions.shape, (x_train.shape[0], 2))

    def test_vector_classification_shared_model(self):
        # Test that Sequential models that feature internal updates
        # and internal losses can be shared.
        np.random.seed(1337)
        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=100, test_samples=0, input_shape=(10,), num_classes=2
        )
        y_train = utils.to_categorical(y_train)

        base_model = test_utils.get_model_from_layers(
            [
                keras.layers.Dense(
                    16,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(1e-5),
                    bias_regularizer=keras.regularizers.l2(1e-5),
                ),
                keras.layers.BatchNormalization(),
            ],
            input_shape=x_train.shape[1:],
        )
        x = keras.layers.Input(x_train.shape[1:])
        y = base_model(x)
        y = keras.layers.Dense(y_train.shape[-1], activation="softmax")(y)
        model = keras.models.Model(x, y)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.legacy.adam.Adam(0.005),
            metrics=["acc"],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        self.assertLen(model.losses, 2)
        if not tf.executing_eagerly():
            self.assertLen(model.get_updates_for(x), 2)
        history = model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=10,
            validation_data=(x_train, y_train),
            verbose=2,
        )
        self.assertGreater(history.history["val_acc"][-1], 0.7)
        _, val_acc = model.evaluate(x_train, y_train)
        self.assertAlmostEqual(history.history["val_acc"][-1], val_acc)
        predictions = model.predict(x_train)
        self.assertEqual(predictions.shape, (x_train.shape[0], 2))


@test_combinations.run_all_keras_modes
class SequentialIntegrationTest(KerasIntegrationTest):
    def test_sequential_save_and_pop(self):
        # Test the following sequence of actions:
        # - construct a Sequential model and train it
        # - save it
        # - load it
        # - pop its last layer and add a new layer instead
        # - continue training
        np.random.seed(1337)
        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=100, test_samples=0, input_shape=(10,), num_classes=2
        )
        y_train = utils.to_categorical(y_train)
        model = keras.Sequential(
            [
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(y_train.shape[-1], activation="softmax"),
            ]
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.legacy.adam.Adam(0.005),
            metrics=["acc"],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(
            x_train,
            y_train,
            epochs=1,
            batch_size=10,
            validation_data=(x_train, y_train),
            verbose=2,
        )
        model = self._save_and_reload_model(model)

        model.pop()
        model.add(keras.layers.Dense(y_train.shape[-1], activation="softmax"))

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.legacy.adam.Adam(0.005),
            metrics=["acc"],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=10,
            validation_data=(x_train, y_train),
            verbose=2,
        )
        self.assertGreater(history.history["val_acc"][-1], 0.7)
        model = self._save_and_reload_model(model)
        _, val_acc = model.evaluate(x_train, y_train)
        self.assertAlmostEqual(history.history["val_acc"][-1], val_acc)
        predictions = model.predict(x_train)
        self.assertEqual(predictions.shape, (x_train.shape[0], 2))


# See b/122473407
@test_combinations.run_all_keras_modes(always_skip_v1=True)
class TimeseriesClassificationIntegrationTest(test_combinations.TestCase):
    @test_combinations.run_with_all_model_types
    def test_timeseries_classification(self):
        np.random.seed(1337)
        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=100,
            test_samples=0,
            input_shape=(4, 10),
            num_classes=2,
        )
        y_train = utils.to_categorical(y_train)

        layers = [
            keras.layers.LSTM(5, return_sequences=True),
            keras.layers.GRU(y_train.shape[-1], activation="softmax"),
        ]
        model = test_utils.get_model_from_layers(
            layers, input_shape=x_train.shape[1:]
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.legacy.adam.Adam(0.005),
            metrics=["acc"],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=15,
            batch_size=10,
            validation_data=(x_train, y_train),
            verbose=2,
        )
        self.assertGreater(history.history["val_acc"][-1], 0.7)
        _, val_acc = model.evaluate(x_train, y_train)
        self.assertAlmostEqual(history.history["val_acc"][-1], val_acc)
        predictions = model.predict(x_train)
        self.assertEqual(predictions.shape, (x_train.shape[0], 2))

    def test_timeseries_classification_sequential_tf_rnn(self):
        np.random.seed(1337)
        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=100,
            test_samples=0,
            input_shape=(4, 10),
            num_classes=2,
        )
        y_train = utils.to_categorical(y_train)

        with base_layer.keras_style_scope():
            model = keras.models.Sequential()
            model.add(
                keras.layers.RNN(
                    legacy_cells.LSTMCell(5),
                    return_sequences=True,
                    input_shape=x_train.shape[1:],
                )
            )
            model.add(
                keras.layers.RNN(
                    legacy_cells.GRUCell(
                        y_train.shape[-1],
                        activation="softmax",
                        dtype=tf.float32,
                    )
                )
            )
            model.compile(
                loss="categorical_crossentropy",
                optimizer=keras.optimizers.legacy.adam.Adam(0.005),
                metrics=["acc"],
                run_eagerly=test_utils.should_run_eagerly(),
            )

        history = model.fit(
            x_train,
            y_train,
            epochs=15,
            batch_size=10,
            validation_data=(x_train, y_train),
            verbose=2,
        )
        self.assertGreater(history.history["val_acc"][-1], 0.7)
        _, val_acc = model.evaluate(x_train, y_train)
        self.assertAlmostEqual(history.history["val_acc"][-1], val_acc)
        predictions = model.predict(x_train)
        self.assertEqual(predictions.shape, (x_train.shape[0], 2))


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class ImageClassificationIntegrationTest(test_combinations.TestCase):
    def test_image_classification(self):
        np.random.seed(1337)
        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=100,
            test_samples=0,
            input_shape=(10, 10, 3),
            num_classes=2,
        )
        y_train = utils.to_categorical(y_train)

        layers = [
            keras.layers.Conv2D(4, 3, padding="same", activation="relu"),
            keras.layers.Conv2D(8, 3, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(8, 3, padding="same"),
            keras.layers.Flatten(),
            keras.layers.Dense(y_train.shape[-1], activation="softmax"),
        ]
        model = test_utils.get_model_from_layers(
            layers, input_shape=x_train.shape[1:]
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.legacy.adam.Adam(0.005),
            metrics=["acc"],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=10,
            validation_data=(x_train, y_train),
            verbose=2,
        )
        self.assertGreater(history.history["val_acc"][-1], 0.7)
        _, val_acc = model.evaluate(x_train, y_train)
        self.assertAlmostEqual(history.history["val_acc"][-1], val_acc)
        predictions = model.predict(x_train)
        self.assertEqual(predictions.shape, (x_train.shape[0], 2))


@test_combinations.run_all_keras_modes
class ActivationV2IntegrationTest(test_combinations.TestCase):
    """Tests activation function V2 in model exporting and loading.

    This test is to verify in TF 2.x, when 'tf.nn.softmax' is used as an
    activation function, its model exporting and loading work as expected.
    Check b/123041942 for details.
    """

    def test_serialization_v2_model(self):
        np.random.seed(1337)
        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=100, test_samples=0, input_shape=(10,), num_classes=2
        )
        y_train = utils.to_categorical(y_train)

        model = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=x_train.shape[1:]),
                keras.layers.Dense(10, activation=tf.nn.relu),
                # To mimic 'tf.nn.softmax' used in TF 2.x.
                keras.layers.Dense(
                    y_train.shape[-1], activation=tf.math.softmax
                ),
            ]
        )

        # Check if 'softmax' is in model.get_config().
        last_layer_activation = model.get_layer(index=2).get_config()[
            "activation"
        ]
        self.assertEqual(last_layer_activation, "softmax")

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.legacy.adam.Adam(0.005),
            metrics=["accuracy"],
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.fit(
            x_train,
            y_train,
            epochs=2,
            batch_size=10,
            validation_data=(x_train, y_train),
            verbose=2,
        )

        output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
        model.save(output_path, save_format="tf")
        loaded_model = keras.models.load_model(output_path)
        self.assertEqual(model.summary(), loaded_model.summary())


@test_combinations.run_with_all_model_types
@test_utils.run_v2_only
class TokenClassificationIntegrationTest(test_combinations.TestCase):
    """Tests a very simple token classification model.

    The main purpose of this test is to verify that everything works as expected
    when input sequences have variable length, and batches are padded only to
    the maximum length of each batch. This is very common in NLP, and results in
    the sequence dimension varying with each batch step for both the features
    and the labels.
    """

    def test_token_classification(self):
        def densify(x, y):
            return x.to_tensor(), y.to_tensor()

        utils.set_random_seed(1337)
        data = tf.ragged.stack(
            [
                np.random.randint(low=0, high=16, size=random.randint(4, 16))
                for _ in range(100)
            ]
        )
        labels = tf.ragged.stack(
            [np.random.randint(low=0, high=3, size=len(arr)) for arr in data]
        )
        features_dataset = tf.data.Dataset.from_tensor_slices(data)
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
        dataset = dataset.batch(batch_size=10)
        dataset = dataset.map(densify)  # Pads with 0 values by default

        layers = [
            keras.layers.Embedding(16, 4),
            keras.layers.Conv1D(4, 5, padding="same", activation="relu"),
            keras.layers.Conv1D(8, 5, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(3, 5, padding="same", activation="softmax"),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(None,))
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["acc"],
        )
        history = model.fit(
            dataset, epochs=10, validation_data=dataset, verbose=2
        )
        self.assertGreater(history.history["val_acc"][-1], 0.5)
        _, val_acc = model.evaluate(dataset)
        self.assertAlmostEqual(history.history["val_acc"][-1], val_acc)
        predictions = model.predict(dataset)
        self.assertIsInstance(predictions, tf.RaggedTensor)
        self.assertEqual(predictions.shape[0], len(dataset) * 10)
        self.assertEqual(predictions.shape[-1], 3)


if __name__ == "__main__":
    tf.test.main()
