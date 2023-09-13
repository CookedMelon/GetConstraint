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
"""Tests for Keras TF utils."""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.testing_infra import test_combinations
from keras.utils import tf_utils

try:
    import attr
except ImportError:
    attr = None


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class TestIsSymbolicTensor(tf.test.TestCase, parameterized.TestCase):
    def test_default_behavior(self):
        if tf.executing_eagerly():
            self.assertFalse(
                tf_utils.is_symbolic_tensor(
                    tf.Variable(name="blah", initial_value=0.0)
                )
            )
            self.assertFalse(
                tf_utils.is_symbolic_tensor(tf.convert_to_tensor(0.0))
            )
            self.assertFalse(
                tf_utils.is_symbolic_tensor(
                    tf.SparseTensor(
                        indices=[[0, 0], [1, 2]],
                        values=[1, 2],
                        dense_shape=[3, 4],
                    )
                )
            )
        else:
            self.assertTrue(
                tf_utils.is_symbolic_tensor(
                    tf.Variable(name="blah", initial_value=0.0)
                )
            )
            self.assertTrue(
                tf_utils.is_symbolic_tensor(tf.convert_to_tensor(0.0))
            )
            self.assertTrue(
                tf_utils.is_symbolic_tensor(
                    tf.SparseTensor(
                        indices=[[0, 0], [1, 2]],
                        values=[1, 2],
                        dense_shape=[3, 4],
                    )
                )
            )

    def test_works_with_registered(self):
        class CustomClass:
            def value(self):
                return tf.convert_to_tensor(42.0)

        tf.register_tensor_conversion_function(
            CustomClass, lambda value, **_: value.value()
        )

        tf_utils.register_symbolic_tensor_type(CustomClass)

        if tf.executing_eagerly():
            self.assertFalse(
                tf_utils.is_symbolic_tensor(
                    tf.Variable(name="blah", initial_value=0.0)
                )
            )
            self.assertFalse(
                tf_utils.is_symbolic_tensor(tf.convert_to_tensor(0.0))
            )
            self.assertFalse(
                tf_utils.is_symbolic_tensor(
                    tf.SparseTensor(
                        indices=[[0, 0], [1, 2]],
                        values=[1, 2],
                        dense_shape=[3, 4],
                    )
                )
            )
            self.assertFalse(tf_utils.is_symbolic_tensor(CustomClass()))
        else:
            self.assertTrue(
                tf_utils.is_symbolic_tensor(
                    tf.Variable(name="blah", initial_value=0.0)
                )
            )
            self.assertTrue(
                tf_utils.is_symbolic_tensor(tf.convert_to_tensor(0.0))
            )
            self.assertTrue(
                tf_utils.is_symbolic_tensor(
                    tf.SparseTensor(
                        indices=[[0, 0], [1, 2]],
                        values=[1, 2],
                        dense_shape=[3, 4],
                    )
                )
            )
            self.assertTrue(tf_utils.is_symbolic_tensor(CustomClass()))

    def test_enables_nontensor_plumbing(self):
        if tf.executing_eagerly():
            self.skipTest("`compile` functionality changed.")
        # Setup.

        class Foo:
            def __init__(self, input_):
                self._input = input_
                self.value = tf.convert_to_tensor([[42.0]])

            @property
            def dtype(self):
                return self.value.dtype

        tf.register_tensor_conversion_function(
            Foo, lambda x, *args, **kwargs: x.value
        )
        tf_utils.register_symbolic_tensor_type(Foo)

        class PlumbingLayer(keras.layers.Lambda):
            def __init__(self, fn, **kwargs):
                def _fn(*fargs, **fkwargs):
                    d = fn(*fargs, **fkwargs)
                    x = tf.convert_to_tensor(d)
                    d.shape = x.shape
                    d.get_shape = x.get_shape
                    return d, x

                super().__init__(_fn, **kwargs)
                self._enter_dunder_call = False

            def __call__(self, inputs, *args, **kwargs):
                self._enter_dunder_call = True
                d, _ = super().__call__(inputs, *args, **kwargs)
                self._enter_dunder_call = False
                return d

            def call(self, inputs, *args, **kwargs):
                d, v = super().call(inputs, *args, **kwargs)
                if self._enter_dunder_call:
                    return d, v
                return d

        # User-land.
        model = keras.Sequential(
            [
                keras.layers.InputLayer((1,)),
                PlumbingLayer(Foo),  # Makes a `Foo` object.
            ]
        )
        # Let's ensure Keras graph history is preserved by composing the models.
        model = keras.Model(model.inputs, model(model.outputs))
        # Now we instantiate the model and verify we have a `Foo` object, not a
        # `Tensor`.
        y = model(tf.convert_to_tensor([[7.0]]))
        self.assertIsInstance(y, Foo)
        # Confirm that (custom) loss sees `Foo` instance, not Tensor.
        obtained_prediction_box = [None]

        def custom_loss(y_obs, y_pred):
            del y_obs
            obtained_prediction_box[0] = y_pred
            return y_pred

        # Apparently `compile` calls the loss function enough to trigger the
        # side-effect.
        model.compile("SGD", loss=custom_loss)
        self.assertIsInstance(obtained_prediction_box[0], Foo)


class ConvertInnerNodeDataTest(tf.test.TestCase):
    def test_convert_inner_node_data(self):
        data = tf_utils.convert_inner_node_data(
            (
                tf_utils.ListWrapper(["l", 2, 3]),
                tf_utils.ListWrapper(["l", 5, 6]),
            )
        )
        self.assertEqual(data, (["l", 2, 3], ["l", 5, 6]))

        data = tf_utils.convert_inner_node_data(
            ((["l", 2, 3], ["l", 5, 6])), wrap=True
        )
        self.assertTrue(
            all(isinstance(ele, tf_utils.ListWrapper) for ele in data)
        )


class AttrsTest(tf.test.TestCase):
    def test_map_structure_with_atomic_accept_attr(self):
        if attr is None:
            self.skipTest("attr module is unavailable.")

        @attr.s(frozen=True)
        class Foo:

            bar = attr.ib()

        self.assertEqual(
            Foo(2),
            tf_utils.map_structure_with_atomic(
                is_atomic_fn=lambda x: isinstance(x, int),
                map_fn=lambda x: x + 1,
                nested=Foo(1),
            ),
        )


class TestIsRagged(tf.test.TestCase):
    def test_is_ragged_return_true_for_ragged_tensor(self):
        tensor = tf.RaggedTensor.from_row_splits(
            values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8]
        )
        self.assertTrue(tf_utils.is_ragged(tensor))

    def test_is_ragged_return_false_for_list(self):
        tensor = [1.0, 2.0, 3.0]
        self.assertFalse(tf_utils.is_ragged(tensor))


class TestIsSparse(tf.test.TestCase):
    def test_is_sparse_return_true_for_sparse_tensor(self):
        tensor = tf.SparseTensor(
            indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]
        )
        self.assertTrue(tf_utils.is_sparse(tensor))

    def test_is_sparse_return_true_for_sparse_tensor_value(self):
        tensor = tf.compat.v1.SparseTensorValue(
            indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]
        )
        self.assertTrue(tf_utils.is_sparse(tensor))

    def test_is_sparse_return_false_for_list(self):
        tensor = [1.0, 2.0, 3.0]
        self.assertFalse(tf_utils.is_sparse(tensor))


class TestIsExtensionType(tf.test.TestCase):
    def test_is_extension_type_return_true_for_ragged_tensor(self):
        self.assertTrue(
            tf_utils.is_extension_type(tf.ragged.constant([[1, 2], [3]]))
        )

    def test_is_extension_type_return_true_for_sparse_tensor(self):
        self.assertTrue(
            tf_utils.is_extension_type(tf.sparse.from_dense([[1, 2], [3, 4]]))
        )

    def test_is_extension_type_return_false_for_dense_tensor(self):
        self.assertFalse(
            tf_utils.is_extension_type(tf.constant([[1, 2], [3, 4]]))
        )

    def test_is_extension_type_return_false_for_list(self):
        tensor = [1.0, 2.0, 3.0]
        self.assertFalse(tf_utils.is_extension_type(tensor))


class TestIsTensorOrExtensionType(tf.test.TestCase):
    def test_is_tensor_or_extension_type_return_true_for_ragged_tensor(self):
        self.assertTrue(
            tf_utils.is_tensor_or_extension_type(
                tf.ragged.constant([[1, 2], [3]])
            )
        )

    def test_is_tensor_or_extension_type_return_true_for_sparse_tensor(self):
        self.assertTrue(
            tf_utils.is_tensor_or_extension_type(
                tf.sparse.from_dense([[1, 2], [3, 4]])
            )
        )

    def test_is_tensor_or_extension_type_return_true_for_dense_tensor(self):
        self.assertTrue(
            tf_utils.is_tensor_or_extension_type(tf.constant([[1, 2], [3, 4]]))
        )

    def test_is_tensor_or_extension_type_return_true_for_custom_ext_types(self):
        class DummyExtensionType(tf.experimental.ExtensionType):
            ...

        self.assertTrue(
            tf_utils.is_tensor_or_extension_type(DummyExtensionType())
        )

    def test_is_tensor_or_extension_type_return_false_for_list(self):
        self.assertFalse(tf_utils.is_tensor_or_extension_type([1.0, 2.0, 3.0]))


@test_combinations.generate(test_combinations.combine(mode=["eager"]))
class TestConvertVariablesToTensors(tf.test.TestCase):
    def test_convert_variables_to_tensors(self):
        x = tf.Variable([1.0])
        result = tf_utils.convert_variables_to_tensors(x)
        self.assertIsInstance(result, tf.Tensor)
        self.assertAllEqual(result, [1.0])

    def test_convert_variables_in_list_to_tensors(self):
        x = [tf.Variable([1.0]), tf.constant([2.0])]
        result = tf_utils.convert_variables_to_tensors(x)
        self.assertLen(result, 2)
        self.assertIsInstance(result[0], tf.Tensor)
        self.assertAllEqual(result[0], [1.0])
        self.assertIs(result[1], x[1])

    def test_convert_variables_in_composite_tensor_to_tensors(self):
        class Spec(tf.TypeSpec):
            value_type = property(lambda self: CompositeVariable)

            def _serialize(self):
                pass

            def _component_specs(self):
                pass

            def _to_components(self, value):
                return value.variables

            def _from_components(self, variable_list):
                return CompositeVariable(variable_list)

        class CompositeVariable(tf.__internal__.CompositeTensor):
            def __init__(self, variable_list):
                self.variables = variable_list

            @property
            def _type_spec(self):
                return Spec()

            def _convert_variables_to_tensors(self):
                self.variables = tf.nest.map_structure(
                    tf_utils.convert_variables_to_tensors, self.variables
                )
                return self

        cv = CompositeVariable([tf.Variable([1.0])])
        self.assertIsInstance(cv.variables[0], tf.Variable)
        result = tf_utils.convert_variables_to_tensors(cv)
        self.assertLen(result.variables, 1)
        self.assertIsInstance(result.variables[0], tf.Tensor)
        self.assertAllEqual(result.variables[0], [1.0])


class TestRandomSeedSetting(tf.test.TestCase):
    def test_seeds(self):
        if not tf.__internal__.tf2.enabled():
            self.skipTest("set_random_seed() is only expected to work in tf2.")

        def get_model_output():
            model = keras.Sequential(
                [
                    keras.layers.Dense(10),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(10),
                ]
            )
            x = np.random.random((32, 10)).astype("float32")
            ds = tf.data.Dataset.from_tensor_slices(x).shuffle(32).batch(16)
            return model.predict(ds)

        tf_utils.set_random_seed(42)
        y1 = get_model_output()
        tf_utils.set_random_seed(42)
        y2 = get_model_output()
        self.assertAllClose(y1, y2, atol=1e-6)


class CustomTypeSpec(tf.TypeSpec):
    """Stubbed-out custom type spec, for testing."""

    def __init__(self, shape, dtype):
        self.shape = tf.TensorShape(shape)
        self.dtype = tf.dtypes.as_dtype(dtype)

    def with_shape(self, new_shape):
        return CustomTypeSpec(new_shape, self.dtype)

    # Stub implementations for all the TypeSpec methods:
    value_type = None
    _to_components = lambda self, value: None
    _from_components = lambda self, components: None
    _component_specs = property(lambda self: None)
    _serialize = lambda self: (self.shape, self.dtype)


class TestGetTensorSpec(parameterized.TestCase):
    @parameterized.parameters(
        [
            (lambda: tf.constant([[1, 2]]), [1, 2]),
            (tf.TensorSpec([8, 3], tf.int32), [8, 3]),
            (tf.TensorSpec([8], tf.int32), [8]),
            (tf.TensorSpec([], tf.int32), []),
            (tf.TensorSpec(None, tf.int32), None),
            (tf.RaggedTensorSpec([8, 3], tf.int32), [8, 3]),
            (tf.SparseTensorSpec([8, 3], tf.int32), [8, 3]),
        ]
    )
    def test_without_dynamic_batch(self, t, expected_shape):
        if callable(t):
            t = t()
        result = tf_utils.get_tensor_spec(t)
        self.assertTrue(result.is_compatible_with(t))
        if expected_shape is None:
            self.assertIsNone(result.shape.rank)
        else:
            self.assertEqual(result.shape.as_list(), expected_shape)

    @parameterized.parameters(
        [
            (lambda: tf.constant([[1, 2]]), [None, 2]),
            (tf.TensorSpec([8, 3], tf.int32), [None, 3]),
            (tf.TensorSpec([8], tf.int32), [None]),
            (tf.TensorSpec([], tf.int32), []),
            (tf.TensorSpec(None, tf.int32), None),
            (tf.RaggedTensorSpec([8, 3], tf.int32), [None, 3]),
            (tf.SparseTensorSpec([8, 3], tf.int32), [None, 3]),
        ]
    )
    def test_with_dynamic_batch(self, t, expected_shape):
        if callable(t):
            t = t()
        result = tf_utils.get_tensor_spec(t, True)
        self.assertTrue(result.is_compatible_with(t))
        if expected_shape is None:
            self.assertIsNone(result.shape.rank)
        else:
            self.assertEqual(result.shape.as_list(), expected_shape)

    def test_with_keras_tensor_with_ragged_spec(self):
        t = keras.engine.keras_tensor.KerasTensor(
            tf.RaggedTensorSpec(shape=(None, None, 1))
        )
        self.assertIsInstance(tf_utils.get_tensor_spec(t), tf.RaggedTensorSpec)


class TestSyncToNumpyOrPythonType(parameterized.TestCase):
    @parameterized.parameters(
        [
            (0.5,),
            (b"string value",),
        ]
    )
    def test_types(self, value):
        if not tf.executing_eagerly():
            self.skipTest("`sync_to_numpy_or_python_type` only works in eager")
        tensor = tf.constant(value)

        self.assertEqual(tf_utils.sync_to_numpy_or_python_type(tensor), value)


class TestCanJitCompile(tf.test.TestCase):
    def test_darwin_arm_xla(self):
        with patch("platform.processor", MagicMock(return_value="arm")):
            with patch("platform.system", MagicMock(return_value="Darwin")):
                self.assertFalse(tf_utils.can_jit_compile())

    def test_linux_xla(self):
        with patch("platform.system", MagicMock(return_value="Linux")):
            self.assertTrue(tf_utils.can_jit_compile())


if __name__ == "__main__":
    tf.test.main()
