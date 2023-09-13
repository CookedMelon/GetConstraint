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
"""Tests specific to Feature Columns integration."""

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras import metrics as metrics_module
from keras.feature_column import dense_features as df
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import np_utils


class TestDNNModel(keras.models.Model):
    def __init__(self, feature_columns, units, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._input_layer = df.DenseFeatures(
            feature_columns, name="input_layer"
        )
        self._dense_layer = keras.layers.Dense(units, name="dense_layer")

    def call(self, features):
        net = self._input_layer(features)
        net = self._dense_layer(net)
        return net


class FeatureColumnsIntegrationTest(test_combinations.TestCase):
    """Most Sequential model API tests are covered in `training_test.py`."""

    @test_combinations.run_all_keras_modes
    def test_sequential_model(self):
        columns = [tf.feature_column.numeric_column("a")]
        model = keras.models.Sequential(
            [
                df.DenseFeatures(columns),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(20, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = {"a": np.random.random((10, 1))}
        y = np.random.randint(20, size=(10, 1))
        y = np_utils.to_categorical(y, num_classes=20)
        model.fit(x, y, epochs=1, batch_size=5)
        model.fit(x, y, epochs=1, batch_size=5)
        model.evaluate(x, y, batch_size=5)
        model.predict(x, batch_size=5)

    @test_combinations.run_all_keras_modes
    def test_sequential_model_with_ds_input(self):
        columns = [tf.feature_column.numeric_column("a")]
        model = keras.models.Sequential(
            [
                df.DenseFeatures(columns),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(20, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        y = np.random.randint(20, size=(100, 1))
        y = np_utils.to_categorical(y, num_classes=20)
        x = {"a": np.random.random((100, 1))}
        ds1 = tf.data.Dataset.from_tensor_slices(x)
        ds2 = tf.data.Dataset.from_tensor_slices(y)
        ds = tf.data.Dataset.zip((ds1, ds2)).batch(5)
        model.fit(ds, steps_per_epoch=1)
        model.fit(ds, steps_per_epoch=1)
        model.evaluate(ds, steps=1)
        model.predict(ds, steps=1)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_sequential_model_with_crossed_column(self):
        feature_columns = []
        age_buckets = tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column("age"),
            boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        )
        feature_columns.append(age_buckets)

        # indicator cols
        thal = tf.feature_column.categorical_column_with_vocabulary_list(
            "thal", ["fixed", "normal", "reversible"]
        )

        crossed_feature = tf.feature_column.crossed_column(
            [age_buckets, thal], hash_bucket_size=1000
        )
        crossed_feature = tf.feature_column.indicator_column(crossed_feature)
        feature_columns.append(crossed_feature)

        feature_layer = df.DenseFeatures(feature_columns)

        model = keras.models.Sequential(
            [
                feature_layer,
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        age_data = np.random.randint(10, 100, size=100)
        thal_data = np.random.choice(
            ["fixed", "normal", "reversible"], size=100
        )
        inp_x = {"age": age_data, "thal": thal_data}
        inp_y = np.random.randint(0, 1, size=100)
        ds = tf.data.Dataset.from_tensor_slices((inp_x, inp_y)).batch(5)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(ds, epochs=1)
        model.fit(ds, epochs=1)
        model.evaluate(ds)
        model.predict(ds)

    @test_combinations.run_all_keras_modes
    def test_subclassed_model_with_feature_columns(self):
        col_a = tf.feature_column.numeric_column("a")
        col_b = tf.feature_column.numeric_column("b")

        dnn_model = TestDNNModel([col_a, col_b], 20)

        dnn_model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        x = {"a": np.random.random((10, 1)), "b": np.random.random((10, 1))}
        y = np.random.randint(20, size=(10, 1))
        y = np_utils.to_categorical(y, num_classes=20)
        dnn_model.fit(x=x, y=y, epochs=1, batch_size=5)
        dnn_model.fit(x=x, y=y, epochs=1, batch_size=5)
        dnn_model.evaluate(x=x, y=y, batch_size=5)
        dnn_model.predict(x=x, batch_size=5)

    @test_combinations.run_all_keras_modes
    def test_subclassed_model_with_feature_columns_with_ds_input(self):
        col_a = tf.feature_column.numeric_column("a")
        col_b = tf.feature_column.numeric_column("b")

        dnn_model = TestDNNModel([col_a, col_b], 20)

        dnn_model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            run_eagerly=test_utils.should_run_eagerly(),
        )

        y = np.random.randint(20, size=(100, 1))
        y = np_utils.to_categorical(y, num_classes=20)
        x = {"a": np.random.random((100, 1)), "b": np.random.random((100, 1))}
        ds1 = tf.data.Dataset.from_tensor_slices(x)
        ds2 = tf.data.Dataset.from_tensor_slices(y)
        ds = tf.data.Dataset.zip((ds1, ds2)).batch(5)
        dnn_model.fit(ds, steps_per_epoch=1)
        dnn_model.fit(ds, steps_per_epoch=1)
        dnn_model.evaluate(ds, steps=1)
        dnn_model.predict(ds, steps=1)

    # TODO(kaftan) seems to throw an error when enabled.
    @test_combinations.run_all_keras_modes
    def DISABLED_test_function_model_feature_layer_input(self):
        col_a = tf.feature_column.numeric_column("a")
        col_b = tf.feature_column.numeric_column("b")

        feature_layer = df.DenseFeatures([col_a, col_b], name="fc")
        dense = keras.layers.Dense(4)

        # This seems problematic.... We probably need something for
        # DenseFeatures the way Input is for InputLayer.
        output = dense(feature_layer)

        model = keras.models.Model([feature_layer], [output])

        optimizer = "rmsprop"
        loss = "mse"
        loss_weights = [1.0, 0.5]
        model.compile(
            optimizer,
            loss,
            metrics=[metrics_module.CategoricalAccuracy(), "mae"],
            loss_weights=loss_weights,
        )

        data = ({"a": np.arange(10), "b": np.arange(10)}, np.arange(10, 20))
        model.fit(*data, epochs=1)

    # TODO(kaftan) seems to throw an error when enabled.
    @test_combinations.run_all_keras_modes
    def DISABLED_test_function_model_multiple_feature_layer_inputs(self):
        col_a = tf.feature_column.numeric_column("a")
        col_b = tf.feature_column.numeric_column("b")
        col_c = tf.feature_column.numeric_column("c")

        fc1 = df.DenseFeatures([col_a, col_b], name="fc1")
        fc2 = df.DenseFeatures([col_b, col_c], name="fc2")
        dense = keras.layers.Dense(4)

        # This seems problematic.... We probably need something for
        # DenseFeatures the way Input is for InputLayer.
        output = dense(fc1) + dense(fc2)

        model = keras.models.Model([fc1, fc2], [output])

        optimizer = "rmsprop"
        loss = "mse"
        loss_weights = [1.0, 0.5]
        model.compile(
            optimizer,
            loss,
            metrics=[metrics_module.CategoricalAccuracy(), "mae"],
            loss_weights=loss_weights,
        )

        data_list = (
            [
                {"a": np.arange(10), "b": np.arange(10)},
                {"b": np.arange(10), "c": np.arange(10)},
            ],
            np.arange(10, 100),
        )
        model.fit(*data_list, epochs=1)

        data_bloated_list = (
            [
                {"a": np.arange(10), "b": np.arange(10), "c": np.arange(10)},
                {"a": np.arange(10), "b": np.arange(10), "c": np.arange(10)},
            ],
            np.arange(10, 100),
        )
        model.fit(*data_bloated_list, epochs=1)

        data_dict = (
            {
                "fc1": {"a": np.arange(10), "b": np.arange(10)},
                "fc2": {"b": np.arange(10), "c": np.arange(10)},
            },
            np.arange(10, 100),
        )
        model.fit(*data_dict, epochs=1)

        data_bloated_dict = (
            {
                "fc1": {
                    "a": np.arange(10),
                    "b": np.arange(10),
                    "c": np.arange(10),
                },
                "fc2": {
                    "a": np.arange(10),
                    "b": np.arange(10),
                    "c": np.arange(10),
                },
            },
            np.arange(10, 100),
        )
        model.fit(*data_bloated_dict, epochs=1)

    @test_combinations.run_all_keras_modes
    def test_string_input(self):
        x = {
            "age": np.random.random((1024, 1)),
            "cabin": np.array(["a"] * 1024),
        }
        y = np.random.randint(2, size=(1024, 1))
        ds1 = tf.data.Dataset.from_tensor_slices(x)
        ds2 = tf.data.Dataset.from_tensor_slices(y)
        dataset = tf.data.Dataset.zip((ds1, ds2)).batch(4)
        categorical_cols = [
            tf.feature_column.categorical_column_with_hash_bucket("cabin", 10)
        ]
        feature_cols = [tf.feature_column.numeric_column("age")] + [
            tf.feature_column.indicator_column(cc) for cc in categorical_cols
        ]
        layers = [
            df.DenseFeatures(feature_cols),
            keras.layers.Dense(128),
            keras.layers.Dense(1),
        ]

        model = keras.models.Sequential(layers)
        model.compile(optimizer="sgd", loss=keras.losses.BinaryCrossentropy())
        model.fit(dataset)


if __name__ == "__main__":
    tf.test.main()
