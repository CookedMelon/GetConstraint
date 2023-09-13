# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.layers.base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import backend
from keras.engine import base_layer as keras_base_layer
from keras.engine import input_spec
from keras.legacy_tf_layers import base as base_tf_layers
from keras.legacy_tf_layers import core as core_tf_layers
from keras.testing_infra import test_combinations


class BaseLayerTest(tf.test.TestCase, parameterized.TestCase):
    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testLayerProperties(self):
        layer = base_tf_layers.Layer(name="my_layer")
        self.assertEqual(layer.variables, [])
        self.assertEqual(layer.trainable_variables, [])
        self.assertEqual(layer.non_trainable_variables, [])
        if not tf.executing_eagerly():
            # updates, losses only supported in GRAPH mode
            self.assertEqual(layer.updates, [])
            self.assertEqual(layer.losses, [])
        self.assertEqual(layer.built, False)
        layer = base_tf_layers.Layer(name="my_layer", trainable=False)
        self.assertEqual(layer.trainable, False)

        # Assert that the layer was not instrumented as a Keras layer
        self.assertFalse(layer._instrumented_keras_api)

        # Assert this was instrumented as a legacy layer
        self.assertTrue(
            keras_base_layer.keras_api_gauge.get_cell("legacy_layer").value()
        )
        keras_base_layer.keras_api_gauge.get_cell("legacy_layer").set(False)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testInt64Layer(self):
        layer = base_tf_layers.Layer(name="my_layer", dtype="int64")
        layer.add_weight("my_var", [2, 2])
        self.assertEqual(layer.name, "my_layer")

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testKerasStyleAddWeight(self):
        keras_layer = keras_base_layer.Layer(name="keras_layer")
        with backend.name_scope("foo"):
            keras_variable = keras_layer.add_weight(
                "my_var", [2, 2], initializer=tf.compat.v1.zeros_initializer()
            )
        self.assertEqual(keras_variable.name, "foo/my_var:0")

        with backend.name_scope("baz"):
            old_style_layer = base_tf_layers.Layer(name="my_layer")
            # Test basic variable creation.
            variable = old_style_layer.add_weight(
                "my_var", [2, 2], initializer=tf.compat.v1.zeros_initializer()
            )
        self.assertEqual(variable.name, "my_layer/my_var:0")

        with base_tf_layers.keras_style_scope():
            layer = base_tf_layers.Layer(name="my_layer")
        # Assert that the layer was not instrumented as a Keras layer
        self.assertFalse(layer._instrumented_keras_api)
        # Test basic variable creation.
        with backend.name_scope("bar"):
            variable = layer.add_weight(
                "my_var", [2, 2], initializer=tf.compat.v1.zeros_initializer()
            )
        self.assertEqual(variable.name, "bar/my_var:0")

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testAddWeight(self):
        layer = base_tf_layers.Layer(name="my_layer")

        # Test basic variable creation.
        variable = layer.add_weight(
            "my_var", [2, 2], initializer=tf.compat.v1.zeros_initializer()
        )
        self.assertEqual(variable.name, "my_layer/my_var:0")
        self.assertEqual(layer.variables, [variable])
        self.assertEqual(layer.trainable_variables, [variable])
        self.assertEqual(layer.non_trainable_variables, [])
        if not tf.executing_eagerly():
            self.assertEqual(
                layer.variables,
                tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
                ),
            )

        # Test non-trainable variable creation.
        # layer.add_variable should work even outside `build` and `call`.
        variable_2 = layer.add_weight(
            "non_trainable_var",
            [2, 2],
            initializer=tf.compat.v1.zeros_initializer(),
            trainable=False,
        )
        self.assertEqual(layer.variables, [variable, variable_2])
        self.assertEqual(layer.trainable_variables, [variable])
        self.assertEqual(layer.non_trainable_variables, [variable_2])

        if not tf.executing_eagerly():
            self.assertEqual(
                len(
                    tf.compat.v1.get_collection(
                        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
                    )
                ),
                1,
            )

        regularizer = lambda x: tf.reduce_sum(x) * 1e-3
        _ = layer.add_weight(
            "reg_var",
            [2, 2],
            initializer=tf.compat.v1.zeros_initializer(),
            regularizer=regularizer,
        )
        self.assertEqual(len(layer.losses), 1)

        added_variable = [False]

        # Test that sync `ON_READ` variables are defaulted to be non-trainable.
        variable_3 = layer.add_weight(
            "sync_on_read_var",
            [2, 2],
            initializer=tf.compat.v1.zeros_initializer(),
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.compat.v1.VariableAggregation.SUM,
        )
        self.assertEqual(
            layer.non_trainable_variables, [variable_2, variable_3]
        )

        @tf.function
        def function_adds_weight():
            if not added_variable[0]:
                layer.add_weight(
                    "reg_var_from_function",
                    [2, 2],
                    initializer=tf.compat.v1.zeros_initializer(),
                    regularizer=regularizer,
                )
                added_variable[0] = True

        function_adds_weight()
        self.assertEqual(len(layer.losses), 2)

    def testInvalidTrainableSynchronizationCombination(self):
        layer = base_tf_layers.Layer(name="my_layer")

        with self.assertRaisesRegex(
            ValueError,
            "Synchronization value can be set to "
            "VariableSynchronization.ON_READ only for non-trainable variables. "
            "You have specified trainable=True and "
            "synchronization=VariableSynchronization.ON_READ.",
        ):
            _ = layer.add_weight(
                "v",
                [2, 2],
                initializer=tf.compat.v1.zeros_initializer(),
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=True,
            )

    def testReusePartitionedVariablesAndRegularizers(self):
        with tf.Graph().as_default():
            regularizer = lambda x: tf.reduce_sum(x) * 1e-3
            partitioner = tf.compat.v1.fixed_size_partitioner(3)
            for reuse in [False, True]:
                with tf.compat.v1.variable_scope(
                    tf.compat.v1.get_variable_scope(),
                    partitioner=partitioner,
                    reuse=reuse,
                ):
                    layer = base_tf_layers.Layer(name="my_layer")
                    _ = layer.add_weight(
                        "reg_part_var",
                        [4, 4],
                        initializer=tf.compat.v1.zeros_initializer(),
                        regularizer=regularizer,
                    )
            self.assertEqual(
                len(
                    tf.compat.v1.get_collection(
                        tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
                    )
                ),
                3,
            )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testCall(self):
        class MyLayer(base_tf_layers.Layer):
            def call(self, inputs):
                return tf.square(inputs)

        layer = MyLayer(name="my_layer")
        inputs = tf.random.uniform((5,), seed=1)
        outputs = layer(inputs)
        self.assertEqual(layer.built, True)
        if not tf.executing_eagerly():
            # op is only supported in GRAPH mode
            self.assertEqual(outputs.op.name, "my_layer/Square")

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testDeepCopy(self):
        class MyLayer(base_tf_layers.Layer):
            def call(self, inputs):
                return tf.square(inputs)

        layer = MyLayer(name="my_layer")
        layer._private_tensor = tf.random.uniform(())
        inputs = tf.random.uniform((5,), seed=1)
        outputs = layer(inputs)
        self.assertEqual(layer.built, True)
        if not tf.executing_eagerly():
            # op only supported in GRAPH mode.
            self.assertEqual(outputs.op.name, "my_layer/Square")

        layer_copy = copy.deepcopy(layer)
        self.assertEqual(layer_copy.name, layer.name)
        self.assertEqual(layer_copy._scope.name, layer._scope.name)
        self.assertEqual(layer_copy._private_tensor, layer._private_tensor)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testScopeNaming(self):
        class PrivateLayer(base_tf_layers.Layer):
            def call(self, inputs):
                return inputs

        inputs = tf.random.uniform((5,))
        default_layer = PrivateLayer()
        _ = default_layer(inputs)
        self.assertEqual(default_layer._scope.name, "private_layer")
        default_layer1 = PrivateLayer()
        default_layer1(inputs)
        self.assertEqual(default_layer1._scope.name, "private_layer_1")
        my_layer = PrivateLayer(name="my_layer")
        my_layer(inputs)
        self.assertEqual(my_layer._scope.name, "my_layer")
        my_layer1 = PrivateLayer(name="my_layer")
        my_layer1(inputs)
        self.assertEqual(my_layer1._scope.name, "my_layer_1")
        my_layer2 = PrivateLayer(name="my_layer")
        my_layer2(inputs)
        self.assertEqual(my_layer2._scope.name, "my_layer_2")
        # Name scope shouldn't affect names.
        with backend.name_scope("some_name_scope"):
            default_layer2 = PrivateLayer()
            default_layer2(inputs)
            self.assertEqual(default_layer2._scope.name, "private_layer_2")
            my_layer3 = PrivateLayer(name="my_layer")
            my_layer3(inputs)
            self.assertEqual(my_layer3._scope.name, "my_layer_3")
            other_layer = PrivateLayer(name="other_layer")
            other_layer(inputs)
            self.assertEqual(other_layer._scope.name, "other_layer")
        # Variable scope gets added to scope names.
        with tf.compat.v1.variable_scope("var_scope"):
            default_layer_scoped = PrivateLayer()
            default_layer_scoped(inputs)
            self.assertEqual(
                default_layer_scoped._scope.name, "var_scope/private_layer"
            )
            my_layer_scoped = PrivateLayer(name="my_layer")
            my_layer_scoped(inputs)
            self.assertEqual(my_layer_scoped._scope.name, "var_scope/my_layer")
            my_layer_scoped1 = PrivateLayer(name="my_layer")
            my_layer_scoped1(inputs)
            self.assertEqual(
                my_layer_scoped1._scope.name, "var_scope/my_layer_1"
            )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testInputSpecNdimCheck(self):
        class CustomerLayer(base_tf_layers.Layer):
            def __init__(self):
                super().__init__()
                self.input_spec = input_spec.InputSpec(ndim=2)

            def call(self, inputs):
                return inputs

        layer = CustomerLayer()
        with self.assertRaisesRegex(ValueError, r"expected ndim=2"):
            layer(tf.constant([1]))

        # Note that we re-create the layer since in Eager mode, input spec
        # checks only happen on first call.
        # Works
        layer = CustomerLayer()
        layer(tf.constant([[1], [2]]))

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testInputSpecMinNdimCheck(self):
        class CustomLayer(base_tf_layers.Layer):
            def __init__(self):
                super().__init__()
                self.input_spec = input_spec.InputSpec(min_ndim=2)

            def call(self, inputs):
                return inputs

        layer = CustomLayer()
        with self.assertRaisesRegex(ValueError, r"expected min_ndim=2"):
            layer(tf.constant([1]))

        # Works
        layer = CustomLayer()
        layer(tf.constant([[1], [2]]))

        layer = CustomLayer()
        layer(tf.constant([[[1], [2]]]))

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testInputSpecMaxNdimCheck(self):
        class CustomerLayer(base_tf_layers.Layer):
            def __init__(self):
                super().__init__()
                self.input_spec = input_spec.InputSpec(max_ndim=2)

            def call(self, inputs):
                return inputs

        layer = CustomerLayer()
        with self.assertRaisesRegex(ValueError, r"expected max_ndim=2"):
            layer(tf.constant([[[1], [2]]]))

        # Works
        layer = CustomerLayer()
        layer(tf.constant([1]))

        layer = CustomerLayer()
        layer(tf.constant([[1], [2]]))

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testInputSpecDtypeCheck(self):
        class CustomerLayer(base_tf_layers.Layer):
            def __init__(self):
                super().__init__()
                self.input_spec = input_spec.InputSpec(dtype="float32")

            def call(self, inputs):
                return inputs

        layer = CustomerLayer()
        with self.assertRaisesRegex(ValueError, r"expected dtype=float32"):
            layer(tf.constant(1, dtype=tf.int32))

        # Works
        layer = CustomerLayer()
        layer(tf.constant(1.0, dtype=tf.float32))

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testInputSpecAxesCheck(self):
        class CustomerLayer(base_tf_layers.Layer):
            def __init__(self):
                super().__init__()
                self.input_spec = input_spec.InputSpec(axes={-1: 2})

            def call(self, inputs):
                return inputs

        layer = CustomerLayer()
        with self.assertRaisesRegex(ValueError, r"expected axis"):
            layer(tf.constant([1, 2, 3]))

        # Works
        layer = CustomerLayer()
        layer(tf.constant([1, 2]))
        layer = CustomerLayer()
        layer(tf.constant([[1, 2], [3, 4], [5, 6]]))

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testInputSpecShapeCheck(self):
        class CustomerLayer(base_tf_layers.Layer):
            def __init__(self):
                super().__init__()
                self.input_spec = input_spec.InputSpec(shape=(None, 3))

            def call(self, inputs):
                return inputs

        layer = CustomerLayer()
        with self.assertRaisesRegex(ValueError, r"expected shape"):
            layer(tf.constant([[1, 2]]))

        # Works
        layer = CustomerLayer()
        layer(tf.constant([[1, 2, 3], [4, 5, 6]]))

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testNoInputSpec(self):
        class CustomerLayer(base_tf_layers.Layer):
            def __init__(self):
                super().__init__()
                self.input_spec = None

            def call(self, inputs):
                return inputs

        layer = CustomerLayer()

        layer(tf.constant(1))

        # Works
        if not tf.executing_eagerly():
            layer(tf.compat.v1.placeholder("int32"))
            layer(tf.compat.v1.placeholder("int32", shape=(2, 3)))

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_count_params(self):
        dense = core_tf_layers.Dense(16)
        dense.build((None, 4))
        self.assertEqual(dense.count_params(), 16 * 4 + 16)

        dense = core_tf_layers.Dense(16)
        with self.assertRaises(ValueError):
            dense.count_params()

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testDictInputOutput(self):
        class DictLayer(base_tf_layers.Layer):
            def call(self, inputs):
                return {"l" + key: inputs[key] for key in inputs}

        layer = DictLayer()
        if tf.executing_eagerly():
            i1 = tf.constant(3)
            i2 = tf.constant(4.0)
            result = layer({"abel": i1, "ogits": i2})
            self.assertTrue(isinstance(result, dict))
            self.assertEqual(set(["label", "logits"]), set(result.keys()))
            self.assertEqual(3, result["label"].numpy())
            self.assertEqual(4.0, result["logits"].numpy())
        else:
            i1 = tf.compat.v1.placeholder("int32")
            i2 = tf.compat.v1.placeholder("float32")
            result = layer({"abel": i1, "ogits": i2})
            self.assertTrue(isinstance(result, dict))
            self.assertEqual(set(["label", "logits"]), set(result.keys()))

    def testActivityRegularizer(self):
        with tf.Graph().as_default():
            regularizer = tf.reduce_sum
            layer = base_tf_layers.Layer(activity_regularizer=regularizer)
            x = tf.compat.v1.placeholder("int32")
            layer(x)
            self.assertEqual(len(layer.get_losses_for(x)), 1)

    def testNameScopeIsConsistentWithVariableScope(self):
        # GitHub issue 13429.

        class MyLayer(base_tf_layers.Layer):
            def build(self, input_shape):
                self.my_var = self.add_weight("my_var", (), tf.float32)
                self.built = True

            def call(self, inputs):
                return tf.multiply(inputs, self.my_var, name="my_op")

        def _gen_layer(x, name=None):
            layer = MyLayer(name=name)
            out = layer(x)
            return layer, out

        # unnamed layer
        with tf.Graph().as_default():
            x = tf.compat.v1.placeholder(tf.float32, (), "x")
            layer, op = _gen_layer(x)
            layer1, op1 = _gen_layer(op)
            layer2, op2 = _gen_layer(op1)

            self.assertEqual(layer.my_var.name, "my_layer/my_var:0")
            self.assertEqual(op.name, "my_layer/my_op:0")
            self.assertEqual(layer1.my_var.name, "my_layer_1/my_var:0")
            self.assertEqual(op1.name, "my_layer_1/my_op:0")
            self.assertEqual(layer2.my_var.name, "my_layer_2/my_var:0")
            self.assertEqual(op2.name, "my_layer_2/my_op:0")
        # name starts from zero
        with tf.Graph().as_default():
            x = tf.compat.v1.placeholder(tf.float32, (), "x")
            layer, op = _gen_layer(x, name="name")
            layer1, op1 = _gen_layer(op, name="name_1")
            layer2, op2 = _gen_layer(op1, name="name_2")

            self.assertEqual(layer.my_var.name, "name/my_var:0")
            self.assertEqual(op.name, "name/my_op:0")
            self.assertEqual(layer1.my_var.name, "name_1/my_var:0")
            self.assertEqual(op1.name, "name_1/my_op:0")
            self.assertEqual(layer2.my_var.name, "name_2/my_var:0")
            self.assertEqual(op2.name, "name_2/my_op:0")
        # name starts from one
        with tf.Graph().as_default():
            x = tf.compat.v1.placeholder(tf.float32, (), "x")
            layer, op = _gen_layer(x, name="name_1")
            layer1, op1 = _gen_layer(op, name="name_2")
            layer2, op2 = _gen_layer(op1, name="name_3")

            self.assertEqual(layer.my_var.name, "name_1/my_var:0")
            self.assertEqual(op.name, "name_1/my_op:0")
            self.assertEqual(layer1.my_var.name, "name_2/my_var:0")
            self.assertEqual(op1.name, "name_2/my_op:0")
            self.assertEqual(layer2.my_var.name, "name_3/my_var:0")
            self.assertEqual(op2.name, "name_3/my_op:0")

    def testVariablesAreLiftedFromFunctionBuildingGraphs(self):
        class MyLayer(base_tf_layers.Layer):
            def build(self, input_shape):
                self.my_var = self.add_weight("my_var", (), tf.float32)
                self.built = True

            def call(self, inputs):
                return inputs

        outer_graph = tf.compat.v1.get_default_graph()
        function_building_graph = tf.Graph()
        function_building_graph._building_function = True
        with outer_graph.as_default():
            with function_building_graph.as_default():
                layer = MyLayer()
                # Create a variable by invoking build through __call__ and
                # assert that it is both tracked and lifted into the outer
                # graph.
                inputs = tf.compat.v1.placeholder(tf.float32, (), "inputs")
                layer(inputs)
                self.assertEqual(len(layer.variables), 1)
                self.assertEqual(len(layer.trainable_variables), 1)
                self.assertEqual(layer.variables[0].graph, outer_graph)

    def testGetUpdateFor(self):
        class MyLayer(base_tf_layers.Layer):
            def build(self, input_shape):
                self.a = self.add_weight("a", (), tf.float32, trainable=False)
                self.b = self.add_weight("b", (), tf.float32, trainable=False)
                self.add_update(
                    tf.compat.v1.assign_add(self.a, 1.0, name="b_update")
                )
                self.built = True

            def call(self, inputs):
                self.add_update(
                    tf.compat.v1.assign_add(self.a, inputs, name="a_update")
                )
                return inputs + 1

        with tf.Graph().as_default():
            layer = MyLayer()
            inputs = tf.compat.v1.placeholder(tf.float32, (), "inputs")
            intermediate_inputs = inputs + 1
            outputs = layer(intermediate_inputs)

            self.assertEqual(len(layer.updates), 2)
            self.assertEqual(len(layer.get_updates_for(None)), 1)
            self.assertEqual(len(layer.get_updates_for([inputs])), 1)
            self.assertEqual(
                len(layer.get_updates_for([intermediate_inputs])), 1
            )
            self.assertEqual(len(layer.get_updates_for([outputs])), 0)

            # Call same layer on new input, creating one more conditional update
            inputs = tf.compat.v1.placeholder(tf.float32, (), "inputs")
            intermediate_inputs = inputs + 1
            outputs = layer(intermediate_inputs)

            self.assertEqual(len(layer.updates), 3)
            self.assertEqual(len(layer.get_updates_for(None)), 1)
            # Check that we are successfully filtering out irrelevant updates
            self.assertEqual(len(layer.get_updates_for([inputs])), 1)
            self.assertEqual(
                len(layer.get_updates_for([intermediate_inputs])), 1
            )
            self.assertEqual(len(layer.get_updates_for([outputs])), 0)

    def testGetLossesFor(self):
        class MyLayer(base_tf_layers.Layer):
            def build(self, input_shape):
                self.a = self.add_weight("a", (), tf.float32, trainable=False)
                self.b = self.add_weight("b", (), tf.float32, trainable=False)
                self.add_loss(self.a)
                self.built = True

            def call(self, inputs):
                self.add_loss(inputs, inputs=True)
                return inputs + 1

        with tf.Graph().as_default():
            layer = MyLayer()
            inputs = tf.compat.v1.placeholder(tf.float32, (), "inputs")
            intermediate_inputs = inputs + 1
            outputs = layer(intermediate_inputs)

            self.assertEqual(len(layer.losses), 2)
            self.assertEqual(len(layer.get_losses_for(None)), 1)
            self.assertEqual(len(layer.get_losses_for([inputs])), 1)
            self.assertEqual(
                len(layer.get_losses_for([intermediate_inputs])), 1
            )
            self.assertEqual(len(layer.get_losses_for([outputs])), 0)

            # Call same layer on new input, creating one more conditional loss
            inputs = tf.compat.v1.placeholder(tf.float32, (), "inputs")
            intermediate_inputs = inputs + 1
            outputs = layer(intermediate_inputs)

            self.assertEqual(len(layer.losses), 3)
            self.assertEqual(len(layer.get_losses_for(None)), 1)
            # Check that we are successfully filtering out irrelevant losses
            self.assertEqual(len(layer.get_losses_for([inputs])), 1)
            self.assertEqual(
                len(layer.get_losses_for([intermediate_inputs])), 1
            )
            self.assertEqual(len(layer.get_losses_for([outputs])), 0)


class IdentityLayer(base_tf_layers.Layer):
    """A layer returns the identity of it's input."""

    def call(self, inputs):
        return inputs


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class DTypeTest(tf.test.TestCase, parameterized.TestCase):
    def _const(self, dtype):
        return tf.constant(1, dtype=dtype)

    def test_dtype_inferred_from_input(self):
        # Test with Tensor input
        layer = IdentityLayer()
        self.assertIsNone(layer.dtype)
        layer(self._const("float64"))
        self.assertEqual(layer.dtype, "float64")

        # Test with Numpy input
        layer = IdentityLayer()
        self.assertIsNone(layer.dtype)
        layer(np.array(1.0, dtype="float64"))
        self.assertEqual(layer.dtype, "float64")

        # Test with integer input
        layer = IdentityLayer()
        self.assertIsNone(layer.dtype)
        layer(self._const("int32"))
        self.assertEqual(layer.dtype, "int32")

        # Test layer dtype doesn't change when passed a new dtype
        layer = IdentityLayer()
        self.assertIsNone(layer.dtype)
        layer(self._const("float64"))
        self.assertEqual(layer.dtype, "float64")
        layer(self._const("float16"))
        self.assertEqual(layer.dtype, "float64")

        # Test layer dtype inferred from first input
        layer = IdentityLayer()
        layer([self._const("float32"), self._const("float64")])
        self.assertEqual(layer.dtype, "float32")

    def test_passing_dtype_to_constructor(self):
        layer = IdentityLayer(dtype="float64")
        layer(self._const("float32"))
        self.assertEqual(layer.dtype, "float64")

        layer = IdentityLayer(dtype="int32")
        layer(self._const("float32"))
        self.assertEqual(layer.dtype, "int32")

        layer = IdentityLayer(dtype=tf.float64)
        layer(self._const("float32"))
        self.assertEqual(layer.dtype, "float64")

    def test_inputs_not_casted(self):
        layer = IdentityLayer(dtype="float32")
        self.assertEqual(layer(self._const("float64")).dtype, "float64")


if __name__ == "__main__":
    tf.test.main()
