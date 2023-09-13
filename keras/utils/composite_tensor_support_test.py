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
"""Tests for Keras composite tensor support."""

import numpy as np
import scipy.sparse
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.engine import input_layer
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Layer
from keras.layers import core
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


# Define test-only Layer classes to validate passing Sparse and Ragged tensors
# between layers.
class ToDense(Layer):
    """Create a dense (standard) tensor from the given input tensor."""

    def __init__(self, default_value, **kwargs):
        super().__init__(**kwargs)
        self._default_value = default_value

    def call(self, inputs):
        if isinstance(inputs, dict):  # Dicts are no longer flattened.
            # Always a single element in these tests.
            inputs = tf.nest.flatten(inputs)[0]

        if isinstance(inputs, tf.RaggedTensor):
            output = inputs.to_tensor(default_value=self._default_value)
        elif isinstance(inputs, tf.SparseTensor):
            output = tf.sparse.to_dense(
                inputs, default_value=self._default_value
            )
        elif isinstance(inputs, tf.Tensor):
            output = inputs
        else:
            raise TypeError(f"Unexpected tensor type {type(inputs).__name__}")

        # Return a float so that we can compile models with this as the final
        # layer.
        return tf.cast(output, tf.float32)


class ToRagged(Layer):
    """Create a ragged tensor based on a given dense tensor."""

    def __init__(self, padding, ragged_rank=1, **kwargs):
        super().__init__(**kwargs)
        self._padding = padding
        self._ragged_rank = ragged_rank

    def call(self, inputs):
        return tf.RaggedTensor.from_tensor(
            inputs, padding=self._padding, ragged_rank=self._ragged_rank
        )


class ToSparse(Layer):
    """Create a sparse tensor based on a given dense tensor."""

    def call(self, inputs):
        indices = tf.where(tf.not_equal(inputs, 0))
        values = tf.gather_nd(inputs, indices)
        shape = tf.shape(inputs, out_type=tf.int64)
        return tf.SparseTensor(indices, values, dense_shape=shape)


class _SubclassModel(keras.Model):
    """A Keras subclass model."""

    def __init__(self, layers, i_layer=None):
        super().__init__()
        # Note that clone and build doesn't support lists of layers in
        # subclassed models. Adding each layer directly here.
        for i, layer in enumerate(layers):
            setattr(self, self._layer_name_for_i(i), layer)
        self.num_layers = len(layers)
        if i_layer is not None:
            self._set_inputs(i_layer)

    def _layer_name_for_i(self, i):
        return f"layer{i}"

    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(self.num_layers):
            layer = getattr(self, self._layer_name_for_i(i))
            x = layer(x)
        return x


def get_model_from_layers_with_input(
    layers, input_shape=None, input_dtype=None, model_input=None
):
    """Builds a model from a sequence of layers."""
    if model_input is not None and input_shape is not None:
        raise ValueError("Cannot specify a model_input and an input shape.")

    model_type = test_utils.get_model_type()
    if model_type == "subclass":
        return _SubclassModel(layers, model_input)

    if model_type == "sequential":
        model = keras.models.Sequential()
        if model_input is not None:
            model.add(model_input)
        elif input_shape is not None:
            model.add(keras.Input(shape=input_shape, dtype=input_dtype))
        for layer in layers:
            model.add(layer)
        return model

    if model_type == "functional":
        if model_input is not None:
            inputs = model_input
        else:
            if not input_shape:
                raise ValueError(
                    "Cannot create a functional model from layers with no "
                    "input shape."
                )
            inputs = keras.Input(shape=input_shape, dtype=input_dtype)
        outputs = inputs
        for layer in layers:
            outputs = layer(outputs)
        return keras.Model(inputs, outputs)

    raise ValueError(f"Unknown model type {model_type}")


def get_test_mode_kwargs():
    run_eagerly = test_utils.should_run_eagerly()
    return {
        "run_eagerly": run_eagerly,
    }


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class CompositeTensorInternalTest(test_combinations.TestCase):
    def test_internal_ragged_tensors(self):
        # Create a model that accepts an input, converts it to Ragged, and
        # converts the ragged tensor back to a dense tensor.
        layers = [ToRagged(padding=0), ToDense(default_value=-1)]
        model = test_utils.get_model_from_layers(layers, input_shape=(None,))

        # Define some input data with additional padding.
        input_data = np.array([[1, 0, 0], [2, 3, 0]])
        expected_output = np.array([[1, -1], [2, 3]])
        output = model.predict(input_data)
        self.assertAllEqual(expected_output, output)

    def test_internal_sparse_tensors(self):
        # Create a model that accepts an input, converts it to Sparse, and
        # converts the sparse tensor back to a dense tensor.
        layers = [ToSparse(), ToDense(default_value=-1)]
        model = test_utils.get_model_from_layers(layers, input_shape=(None,))

        # Define some input data with additional padding.
        input_data = np.array([[1, 0, 0], [2, 3, 0]])
        expected_output = np.array([[1, -1, -1], [2, 3, -1]])
        output = model.predict(input_data)
        self.assertAllEqual(expected_output, output)

    def test_training_internal_ragged_tensors(self):
        # Create a model that implements y=Mx. This is easy to learn and will
        # demonstrate appropriate gradient passing. (We have to use
        # RaggedTensors for this test, as ToSparse() doesn't support gradient
        # propagation through the layer.) TODO(b/124796939): Investigate this.
        layers = [core.Dense(2), ToRagged(padding=0), ToDense(default_value=-1)]
        model = test_utils.get_model_from_layers(layers, input_shape=(1,))

        input_data = np.random.rand(1024, 1)
        expected_data = np.concatenate(
            (input_data * 3, input_data * 0.5), axis=-1
        )

        model.compile(loss="mse", optimizer="adam", **get_test_mode_kwargs())
        history = model.fit(input_data, expected_data, epochs=10, verbose=0)

        # If the model trained, the loss stored at history[0] should be
        # different than the one stored at history[-1].
        self.assertNotEqual(
            history.history["loss"][-1], history.history["loss"][0]
        )


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class CompositeTensorOutputTest(test_combinations.TestCase):
    def test_ragged_tensor_outputs(self):
        # Create a model that accepts an input, converts it to Ragged, and
        # converts the ragged tensor back to a dense tensor.
        layers = [ToRagged(padding=0)]
        model = test_utils.get_model_from_layers(layers, input_shape=(None,))
        model._run_eagerly = test_utils.should_run_eagerly()

        # Define some input data with additional padding.
        input_data = np.array([[1, 0, 0], [2, 3, 0]])
        output = model.predict(input_data)

        expected_values = [[1], [2, 3]]
        self.assertAllEqual(expected_values, output)

    def test_ragged_tensor_rebatched_outputs(self):
        # Create a model that accepts an input, converts it to Ragged, and
        # converts the ragged tensor back to a dense tensor.
        layers = [ToRagged(padding=0)]
        model = test_utils.get_model_from_layers(layers, input_shape=(None,))
        model._run_eagerly = test_utils.should_run_eagerly()

        # Define some input data with additional padding.
        input_data = np.array([[1, 0, 0], [2, 3, 0], [4, 0, 0], [5, 6, 0]])
        output = model.predict(input_data, batch_size=2)

        expected_values = [[1], [2, 3], [4], [5, 6]]
        self.assertAllEqual(expected_values, output)

    def test_sparse_tensor_outputs(self):
        # Create a model that accepts an input, converts it to Ragged, and
        # converts the ragged tensor back to a dense tensor.
        layers = [ToSparse()]
        model = test_utils.get_model_from_layers(layers, input_shape=(None,))
        model._run_eagerly = test_utils.should_run_eagerly()

        # Define some input data with additional padding.
        input_data = np.array([[1, 0, 0], [2, 3, 0]])
        output = model.predict(input_data)

        expected_indices = np.array([[0, 0], [1, 0], [1, 1]])
        expected_values = np.array([1, 2, 3])
        expected_dense_shape = np.array([2, 3])

        self.assertAllEqual(output.indices, expected_indices)
        self.assertAllEqual(output.values, expected_values)
        self.assertAllEqual(output.dense_shape, expected_dense_shape)

    def test_sparse_tensor_rebatched_outputs(self):
        # Create a model that accepts an input, converts it to Ragged, and
        # converts the ragged tensor back to a dense tensor.
        layers = [ToSparse()]
        model = test_utils.get_model_from_layers(layers, input_shape=(None,))
        model._run_eagerly = test_utils.should_run_eagerly()

        # Define some input data with additional padding.
        input_data = np.array([[1, 0, 0], [2, 3, 0], [4, 0, 0], [5, 6, 0]])
        output = model.predict(input_data, batch_size=2)

        expected_indices = np.array(
            [[0, 0], [1, 0], [1, 1], [2, 0], [3, 0], [3, 1]]
        )
        expected_values = np.array([1, 2, 3, 4, 5, 6])
        expected_dense_shape = np.array([4, 3])

        self.assertAllEqual(output.indices, expected_indices)
        self.assertAllEqual(output.values, expected_values)
        self.assertAllEqual(output.dense_shape, expected_dense_shape)


def get_input_name(use_dict):
    # Define the input name.
    if not use_dict:
        return None  # This is the same as not setting 'name'.
    elif test_utils.get_model_type() == "subclass":
        return "input_1"  # Subclass models don"t support input names.
    else:
        return "test_input_name"


def get_kwargs(use_dataset, action="predict"):
    if use_dataset or not tf.executing_eagerly():
        if action == "fit":
            return {"steps_per_epoch": 1}
        return {"steps": 1}
    else:
        return {"batch_size": 2}


def prepare_inputs(data, use_dict, use_dataset, action, input_name):
    input_data, expected_output = data
    batch_size = input_data.shape[0]
    # Prepare the input data.
    if use_dict:
        input_data = {input_name: input_data}
    if use_dataset:
        if action == "predict":
            input_data = tf.data.Dataset.from_tensor_slices(input_data).batch(
                batch_size
            )
        else:
            input_data = tf.data.Dataset.from_tensor_slices(
                (input_data, expected_output)
            ).batch(batch_size)
            expected_output = None
    return (input_data, expected_output)


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
@parameterized.named_parameters(
    *test_utils.generate_combinations_with_testcase_name(
        use_dict=[True, False],
        use_dataset=[True, False],
        action=["predict", "evaluate", "fit"],
    )
)
class SparseTensorInputTest(test_combinations.TestCase):
    def test_sparse_tensors(self, use_dict, use_dataset, action):
        data = [
            (
                tf.SparseTensor(
                    [[0, 0, 0], [1, 0, 0], [1, 0, 1]], [1, 2, 3], [2, 1, 3]
                ),
                np.array([[[1, -1, -1]], [[2, 3, -1]]]),
            ),
            (
                tf.SparseTensor(
                    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [2, 0, 1]],
                    [5, 6, 7, 8],
                    [3, 1, 4],
                ),
                np.array(
                    [[[5, -1, -1, -1]], [[6, 7, -1, -1]], [[-1, 8, -1, -1]]]
                ),
            ),
        ]
        # Prepare the model to test.
        input_name = get_input_name(use_dict)
        model_input = input_layer.Input(
            shape=(1, None), sparse=True, name=input_name, dtype=tf.int32
        )
        layers = [ToDense(default_value=-1)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )
        model.compile(
            optimizer="sgd",
            loss="mse",
            metrics=["accuracy"],
            **get_test_mode_kwargs(),
        )
        kwargs = get_kwargs(use_dataset, action)

        # Prepare the input data
        for data_element in data:
            input_data, expected_output = prepare_inputs(
                data_element, use_dict, use_dataset, action, input_name
            )
            # Perform the action.
            if action == "predict":
                result = model.predict(input_data, **kwargs)
                self.assertAllEqual(expected_output, result)
            if action == "evaluate":
                result = model.evaluate(input_data, expected_output, **kwargs)
                self.assertAllEqual(1.0, result[-1])
            if action == "fit":
                # TODO(momernick): What's the best way of validating that fit
                # happened?
                _ = model.fit(
                    input_data, expected_output, shuffle=False, **kwargs
                )


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class ScipySparseTensorInputTest(test_combinations.TestCase, tf.test.TestCase):
    def test_sparse_scipy_predict_inputs_via_input_layer_args(self):
        # Create a model that accepts a sparse input and converts the sparse
        # tensor back to a dense tensor. Scipy sparse matrices are limited to
        # 2D, so use a one-dimensional shape; note also that scipy's default
        # dtype is int64.
        model_input = input_layer.Input(shape=(3,), sparse=True, dtype=tf.int64)
        layers = [ToDense(default_value=-1)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )

        input_data = scipy.sparse.coo_matrix(
            ([1, 2, 3], ([0, 1, 1], [0, 0, 1])), shape=[2, 3]
        )
        expected_output = np.array([[1, -1, -1], [2, 3, -1]])
        output = model.predict(input_data, steps=1)
        self.assertAllEqual(expected_output, output)

        input_data_2 = scipy.sparse.coo_matrix(
            ([5, 6, 7, 8], ([0, 1, 1, 2], [0, 0, 1, 1])), shape=[3, 3]
        )
        expected_output_2 = np.array([[5, -1, -1], [6, 7, -1], [-1, 8, -1]])
        output_2 = model.predict(input_data_2, steps=1)
        self.assertAllEqual(expected_output_2, output_2)

    def test_sparse_scipy_eval_inputs(self):
        # Create a model that accepts a sparse input and converts the sparse
        # tensor back to a dense tensor. Scipy sparse matrices are limited to
        # 2D, so use a one-dimensional shape; note also that scipy's default
        # dtype is int64.
        model_input = input_layer.Input(shape=(3,), sparse=True, dtype=tf.int64)
        layers = [ToDense(default_value=-1)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )
        model.compile(optimizer="sgd", loss="mse", metrics=["accuracy"])

        input_data = scipy.sparse.coo_matrix(
            ([1, 2, 3], ([0, 1, 1], [0, 0, 1])), shape=[2, 3]
        )
        expected_output = np.array([[1, -1, -1], [2, 3, -1]])

        output = model.evaluate(input_data, expected_output, steps=1)
        self.assertAllEqual(1.0, output[-1])

        input_data_2 = scipy.sparse.coo_matrix(
            ([5, 6, 7, 8], ([0, 1, 1, 2], [0, 0, 1, 1])), shape=[3, 3]
        )
        expected_output_2 = np.array([[5, -1, -1], [6, 7, -1], [-1, 8, -1]])
        output_2 = model.evaluate(input_data_2, expected_output_2, steps=1)
        self.assertAllEqual(1.0, output_2[-1])

    def test_sparse_scipy_predict_input_dicts_via_input_layer_args(self):
        # Create a model that accepts a sparse input and converts the sparse
        # tensor back to a dense tensor. Scipy sparse matrices are limited to
        # 2D, so use a one-dimensional shape; note also that scipy's default
        # dtype is int64.
        if test_utils.get_model_type() == "subclass":
            input_name = "input_1"  # Subclass models don"t support input names.
        else:
            input_name = "test_input_name"
        model_input = input_layer.Input(
            shape=(3,), sparse=True, name=input_name, dtype=tf.int64
        )
        layers = [ToDense(default_value=-1)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )

        input_data = {
            input_name: scipy.sparse.coo_matrix(
                ([1, 2, 3], ([0, 1, 1], [0, 0, 1])), shape=[2, 3]
            )
        }
        expected_output = np.array([[1, -1, -1], [2, 3, -1]])
        output = model.predict(input_data, steps=1)
        self.assertAllEqual(expected_output, output)

        input_data_2 = {
            input_name: scipy.sparse.coo_matrix(
                ([5, 6, 7, 8], ([0, 1, 1, 2], [0, 0, 1, 1])), shape=[3, 3]
            )
        }
        expected_output_2 = np.array([[5, -1, -1], [6, 7, -1], [-1, 8, -1]])
        output_2 = model.predict(input_data_2, steps=1)
        self.assertAllEqual(expected_output_2, output_2)

    def test_sparse_scipy_eval_input_dicts(self):
        # Create a model that accepts a sparse input and converts the sparse
        # tensor back to a dense tensor. Scipy sparse matrices are limited to
        # 2D, so use a one-dimensional shape; note also that scipy's default
        # dtype is int64.
        if test_utils.get_model_type() == "subclass":
            input_name = "input_1"  # Subclass models don"t support input names.
        else:
            input_name = "test_input_name"
        model_input = input_layer.Input(
            shape=(3,), sparse=True, name=input_name, dtype=tf.int64
        )
        layers = [ToDense(default_value=-1)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )
        model.compile(optimizer="sgd", loss="mse", metrics=["accuracy"])

        input_data = {
            input_name: scipy.sparse.coo_matrix(
                ([1, 2, 3], ([0, 1, 1], [0, 0, 1])), shape=[2, 3]
            )
        }
        expected_output = np.array([[1, -1, -1], [2, 3, -1]])
        output = model.evaluate(input_data, expected_output, steps=1)
        self.assertAllEqual(1.0, output[-1])

        input_data_2 = {
            input_name: scipy.sparse.coo_matrix(
                ([5, 6, 7, 8], ([0, 1, 1, 2], [0, 0, 1, 1])), shape=[3, 3]
            )
        }
        expected_output_2 = np.array([[5, -1, -1], [6, 7, -1], [-1, 8, -1]])
        output_2 = model.evaluate(input_data_2, expected_output_2, steps=1)
        self.assertAllEqual(1.0, output_2[-1])


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
@parameterized.named_parameters(
    *test_utils.generate_combinations_with_testcase_name(
        use_dict=[True, False],
        use_dataset=[True, False],
        action=["predict", "evaluate", "fit"],
    )
)
class RaggedTensorInputTest(test_combinations.TestCase, tf.test.TestCase):
    def test_ragged_input(self, use_dict, use_dataset, action):
        data = [
            (
                tf.ragged.constant([[[1]], [[2, 3]]]),
                np.array([[[1, -1]], [[2, 3]]]),
            )
        ]

        # Prepare the model to test.
        input_name = get_input_name(use_dict)
        model_input = input_layer.Input(
            shape=(None, None),
            ragged=True,
            name=input_name,
            dtype=tf.int32,
            batch_size=2,
        )
        self.assertIsInstance(model_input._type_spec, tf.RaggedTensorSpec)
        self.assertEqual(model_input.shape.as_list(), [2, None, None])
        layers = [ToDense(default_value=-1)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )
        model.compile(
            optimizer="sgd",
            loss="mse",
            metrics=["accuracy"],
            **get_test_mode_kwargs(),
        )

        # Prepare the input data
        for data_element in data:
            input_data, expected_output = prepare_inputs(
                data_element, use_dict, use_dataset, action, input_name
            )
            # Perform the action.
            if action == "predict":
                result = model.predict(input_data)
                self.assertAllEqual(expected_output, result)
            if action == "evaluate":
                result = model.evaluate(input_data, expected_output)
                self.assertAllEqual(1.0, result[-1])
            if action == "fit":
                # TODO(momernick): What's the best way of validating that fit
                # happened?
                _ = model.fit(input_data, expected_output, shuffle=False)


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
@parameterized.named_parameters(
    *test_utils.generate_combinations_with_testcase_name(
        use_dict=[True, False], use_dataset=[True, False]
    )
)
class RaggedTensorInputValidationTest(
    test_combinations.TestCase, tf.test.TestCase
):
    def test_ragged_tensor_input_with_one_none_dimension(
        self, use_dict, use_dataset
    ):
        # Define some input data.
        data = [
            (
                tf.ragged.constant([[[1, 0]], [[2, 3]]], ragged_rank=1),
                np.array([[[1, 0]], [[2, 3]]]),
            )
        ]

        # Prepare the model to test.
        input_shape = (None, 2)  # RaggedTensorInputTest uses (None, None).
        input_name = get_input_name(use_dict)
        model_input = input_layer.Input(
            shape=input_shape, ragged=True, name=input_name, dtype=tf.int32
        )
        layers = [ToDense(default_value=-1)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )
        model.compile(
            optimizer="sgd",
            loss="mse",
            metrics=["accuracy"],
            **get_test_mode_kwargs(),
        )

        for data_element in data:
            input_data, expected_output = prepare_inputs(
                data_element,
                use_dict,
                use_dataset,
                action="predict",
                input_name=input_name,
            )
            result = model.predict(input_data)
            self.assertAllEqual(expected_output, result)

    def test_ragged_tensor_input_with_no_none_dimension(
        self, use_dict, use_dataset
    ):
        # Define some input data.
        data = [
            (
                tf.ragged.constant([[[1, 0]], [[2, 3]]], ragged_rank=0),
                np.array([[[1, 0]], [[2, 3]]]),
            )
        ]

        # Prepare the model to test.
        input_shape = (1, 2)  # RaggedTensorInputTest uses (None, None).
        input_name = get_input_name(use_dict)
        model_input = input_layer.Input(
            shape=input_shape, ragged=True, name=input_name, dtype=tf.int32
        )
        layers = [ToDense(default_value=-1)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )
        model.compile(
            optimizer="sgd",
            loss="mse",
            metrics=["accuracy"],
            **get_test_mode_kwargs(),
        )
        kwargs = get_kwargs(use_dataset)

        for data_element in data:
            input_data, expected_output = prepare_inputs(
                data_element,
                use_dict,
                use_dataset,
                action="predict",
                input_name=input_name,
            )
            result = model.predict(input_data, **kwargs)
            self.assertAllEqual(expected_output, result)


@test_combinations.run_with_all_model_types()
@test_combinations.run_all_keras_modes(always_skip_v1=True)
class CompositeTensorModelPredictTest(test_combinations.TestCase):
    def _normalize_shape(self, shape):
        if not isinstance(shape, tuple):
            shape = tuple(shape.as_list())
        return shape

    def test_sparse_tensor_model_predict(self):
        # Create a model that accepts a sparse input and runs a "Dense" layer on
        # it.
        model_input = input_layer.Input(
            shape=(3,), sparse=True, dtype=tf.float32
        )

        self.assertEqual([None, 3], model_input.shape.as_list())

        layers = [Dense(2)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )

        sparse_input = tf.SparseTensor(
            # A two-row matrix
            indices=[(0, 0), (0, 1), (0, 2), (5, 0), (5, 1), (5, 2)],
            values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dense_shape=(6, 3),
        )

        shape = model(sparse_input).shape
        self.assertEqual((6, 2), self._normalize_shape(shape))

        shape = model.predict(sparse_input, steps=1).shape
        self.assertEqual((6, 2), self._normalize_shape(shape))

    def test_ragged_tensor_model_predict(self):
        # Create a model that accepts a sparse input and runs a "Dense" layer on
        # it.
        model_input = input_layer.Input(shape=(None,), ragged=True)
        self.assertEqual([None, None], model_input.shape.as_list())

        layers = [Embedding(input_dim=7, output_dim=5)]
        model = get_model_from_layers_with_input(
            layers, model_input=model_input
        )

        ragged_input = tf.ragged.constant(
            [
                [1, 2, 3, 4, 5],
                [2, 4],
            ]
        )

        shape = model(ragged_input).shape
        self.assertEqual((2, None, 5), self._normalize_shape(shape))

        shape = model.predict(ragged_input, steps=1).shape
        self.assertEqual((2, None, 5), self._normalize_shape(shape))


if __name__ == "__main__":
    tf.test.main()
