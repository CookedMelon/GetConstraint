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
"""Tests for tf.layers.convolutional."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from keras.legacy_tf_layers import convolutional as conv_layers


class ConvTest(tf.test.TestCase):
    def testInvalidDataFormat(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "data_format"):
            conv_layers.conv2d(images, 32, 3, data_format="invalid")

    def testInvalidStrides(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.conv2d(images, 32, 3, strides=(1, 2, 3))

        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.conv2d(images, 32, 3, strides=None)

    def testInvalidKernelSize(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.conv2d(images, 32, (1, 2, 3))

        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.conv2d(images, 32, None)

    def testCreateConv2D(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4))
        layer = conv_layers.Conv2D(32, [3, 3], activation=tf.nn.relu)
        output = layer(images)
        if not tf.executing_eagerly():
            self.assertEqual(output.op.name, "conv2d/Relu")
        self.assertListEqual(
            output.get_shape().as_list(), [5, height - 2, width - 2, 32]
        )
        self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 4, 32])
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testConv2DFloat16(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4), dtype="float16")
        output = conv_layers.conv2d(images, 32, [3, 3], activation=tf.nn.relu)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height - 2, width - 2, 32]
        )

    def testCreateConv2DIntegerKernelSize(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4))
        layer = conv_layers.Conv2D(32, 3)
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height - 2, width - 2, 32]
        )
        self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 4, 32])
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testCreateConv2DChannelsFirst(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, 4, height, width))
            layer = conv_layers.Conv2D(32, [3, 3], data_format="channels_first")
            output = layer(images)
            self.assertListEqual(
                output.get_shape().as_list(), [5, 32, height - 2, width - 2]
            )
            self.assertListEqual(
                layer.kernel.get_shape().as_list(), [3, 3, 4, 32]
            )
            self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testUnknownInputChannels(self):
        with tf.Graph().as_default():
            images = tf.compat.v1.placeholder(tf.float32, (5, 7, 9, None))
            layer = conv_layers.Conv2D(32, [3, 3], activation=tf.nn.relu)
            with self.assertRaisesRegex(
                ValueError,
                "The channel dimension of the inputs "
                "should be defined. The input_shape received is",
            ):
                _ = layer(images)

            images = tf.compat.v1.placeholder(tf.float32, (5, None, 7, 9))
            layer = conv_layers.Conv2D(32, [3, 3], data_format="channels_first")
            with self.assertRaisesRegex(
                ValueError,
                "The channel dimension of the inputs "
                "should be defined. The input_shape received is",
            ):
                _ = layer(images)

    def testConv2DPaddingSame(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 32), seed=1)
        layer = conv_layers.Conv2D(64, images.get_shape()[1:3], padding="same")
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height, width, 64]
        )

    def testCreateConvWithStrides(self):
        height, width = 6, 8
        # Test strides tuple
        images = tf.random.uniform((5, height, width, 3), seed=1)
        layer = conv_layers.Conv2D(32, [3, 3], strides=(2, 2), padding="same")
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height / 2, width / 2, 32]
        )

        # Test strides integer
        layer = conv_layers.Conv2D(32, [3, 3], strides=2, padding="same")
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height / 2, width / 2, 32]
        )

        # Test unequal strides
        layer = conv_layers.Conv2D(32, [3, 3], strides=(2, 1), padding="same")
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height / 2, width, 32]
        )

    def testCreateConv1D(self):
        width = 7
        data = tf.random.uniform((5, width, 4))
        layer = conv_layers.Conv1D(32, 3, activation=tf.nn.relu)
        output = layer(data)
        if not tf.executing_eagerly():
            self.assertEqual(output.op.name, "conv1d/Relu")
        self.assertListEqual(output.get_shape().as_list(), [5, width - 2, 32])
        self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 4, 32])
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testConv1DFloat16(self):
        width = 7
        data = tf.random.uniform((5, width, 4), dtype="float16")
        output = conv_layers.conv1d(data, 32, 3, activation=tf.nn.relu)
        self.assertListEqual(output.get_shape().as_list(), [5, width - 2, 32])

    def testCreateConv1DChannelsFirst(self):
        with tf.Graph().as_default():
            width = 7
            data = tf.random.uniform((5, 4, width))
            layer = conv_layers.Conv1D(32, 3, data_format="channels_first")
            output = layer(data)
            self.assertListEqual(
                output.get_shape().as_list(), [5, 32, width - 2]
            )
            self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 4, 32])
            self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testUnknownInputChannelsConv1D(self):
        with tf.Graph().as_default():
            data = tf.compat.v1.placeholder(tf.float32, (5, 4, None))
            layer = conv_layers.Conv1D(32, 3, activation=tf.nn.relu)
            with self.assertRaisesRegex(
                ValueError,
                "The channel dimension of the inputs "
                "should be defined. The input_shape received is",
            ):
                _ = layer(data)

            data = tf.compat.v1.placeholder(tf.float32, (5, None, 4))
            layer = conv_layers.Conv1D(32, 3, data_format="channels_first")
            with self.assertRaisesRegex(
                ValueError,
                "The channel dimension of the inputs "
                "should be defined. The input_shape received is",
            ):
                _ = layer(data)

    def testCreateConv3D(self):
        depth, height, width = 6, 7, 9
        volumes = tf.random.uniform((5, depth, height, width, 4))
        layer = conv_layers.Conv3D(32, [3, 3, 3], activation=tf.nn.relu)
        output = layer(volumes)
        if not tf.executing_eagerly():
            self.assertEqual(output.op.name, "conv3d/Relu")
        self.assertListEqual(
            output.get_shape().as_list(),
            [5, depth - 2, height - 2, width - 2, 32],
        )
        self.assertListEqual(
            layer.kernel.get_shape().as_list(), [3, 3, 3, 4, 32]
        )
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testUnknownInputChannelsConv3D(self):
        with tf.Graph().as_default():
            volumes = tf.compat.v1.placeholder(tf.float32, (5, 6, 7, 9, None))
            layer = conv_layers.Conv3D(32, [3, 3, 3], activation=tf.nn.relu)
            with self.assertRaisesRegex(
                ValueError,
                "The channel dimension of the inputs "
                "should be defined. The input_shape received is",
            ):
                _ = layer(volumes)

    def testConv2DKernelRegularizer(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.Conv2D(32, [3, 3], kernel_regularizer=reg)
            layer(images)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testConv2DBiasRegularizer(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.Conv2D(32, [3, 3], bias_regularizer=reg)
            layer(images)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testConv2DNoBias(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4))
        layer = conv_layers.Conv2D(
            32, [3, 3], activation=tf.nn.relu, use_bias=False
        )
        output = layer(images)
        if not tf.executing_eagerly():
            self.assertEqual(output.op.name, "conv2d/Relu")
        self.assertListEqual(
            output.get_shape().as_list(), [5, height - 2, width - 2, 32]
        )
        self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 4, 32])
        self.assertEqual(layer.bias, None)

    def testDilatedConv2D(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4))
        layer = conv_layers.Conv2D(32, [3, 3], dilation_rate=3)
        output = layer(images)
        self.assertListEqual(output.get_shape().as_list(), [5, 1, 3, 32])
        self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 4, 32])
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

        # Test tuple dilation rate
        layer = conv_layers.Conv2D(32, [3, 3], dilation_rate=(1, 3))
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height - 2, 3, 32]
        )

    def testFunctionalConv2DReuse(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 3), seed=1)
            conv_layers.conv2d(images, 32, [3, 3], name="conv1")
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)
            conv_layers.conv2d(images, 32, [3, 3], name="conv1", reuse=True)
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)

    def testFunctionalConv2DReuseFromScope(self):
        with tf.Graph().as_default():
            with tf.compat.v1.variable_scope("scope"):
                height, width = 7, 9
                images = tf.random.uniform((5, height, width, 3), seed=1)
                conv_layers.conv2d(images, 32, [3, 3], name="conv1")
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)
            with tf.compat.v1.variable_scope("scope", reuse=True):
                conv_layers.conv2d(images, 32, [3, 3], name="conv1")
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)

    def testFunctionalConv2DInitializerFromScope(self):
        with tf.Graph().as_default(), self.cached_session():
            with tf.compat.v1.variable_scope(
                "scope", initializer=tf.compat.v1.ones_initializer()
            ):
                height, width = 7, 9
                images = tf.random.uniform((5, height, width, 3), seed=1)
                conv_layers.conv2d(images, 32, [3, 3], name="conv1")
                weights = tf.compat.v1.trainable_variables()
                # Check the names of weights in order.
                self.assertTrue("kernel" in weights[0].name)
                self.assertTrue("bias" in weights[1].name)
                self.evaluate(tf.compat.v1.global_variables_initializer())
                weights = self.evaluate(weights)
                # Check that the kernel weights got initialized to ones (from
                # scope)
                self.assertAllClose(weights[0], np.ones((3, 3, 3, 32)))
                # Check that the bias still got initialized to zeros.
                self.assertAllClose(weights[1], np.zeros((32)))

    def testFunctionalConv2DNoReuse(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 3), seed=1)
            conv_layers.conv2d(images, 32, [3, 3])
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)
            conv_layers.conv2d(images, 32, [3, 3])
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 4)

    def testConstraints(self):
        # Conv1D
        k_constraint = lambda x: x / tf.reduce_sum(x)
        b_constraint = lambda x: x / tf.reduce_max(x)
        conv1d = conv_layers.Conv1D(
            2, 3, kernel_constraint=k_constraint, bias_constraint=b_constraint
        )
        inputs = tf.random.uniform((5, 3, 5), seed=1)
        conv1d(inputs)
        self.assertEqual(conv1d.kernel_constraint, k_constraint)
        self.assertEqual(conv1d.bias_constraint, b_constraint)

        # Conv2D
        k_constraint = lambda x: x / tf.reduce_sum(x)
        b_constraint = lambda x: x / tf.reduce_max(x)
        conv2d = conv_layers.Conv2D(
            2, 3, kernel_constraint=k_constraint, bias_constraint=b_constraint
        )
        inputs = tf.random.uniform((5, 3, 3, 5), seed=1)
        conv2d(inputs)
        self.assertEqual(conv2d.kernel_constraint, k_constraint)
        self.assertEqual(conv2d.bias_constraint, b_constraint)

        # Conv3D
        k_constraint = lambda x: x / tf.reduce_sum(x)
        b_constraint = lambda x: x / tf.reduce_max(x)
        conv3d = conv_layers.Conv3D(
            2, 3, kernel_constraint=k_constraint, bias_constraint=b_constraint
        )
        inputs = tf.random.uniform((5, 3, 3, 3, 5), seed=1)
        conv3d(inputs)
        self.assertEqual(conv3d.kernel_constraint, k_constraint)
        self.assertEqual(conv3d.bias_constraint, b_constraint)

    def testConv3DChannelsFirst(self):
        # Test case for GitHub issue 15655
        with tf.Graph().as_default():
            images = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, 1, 32, 32, 32]
            )
            conv_layers.conv3d(images, 32, 9, data_format="channels_first")


class SeparableConv1DTest(tf.test.TestCase):
    def testInvalidDataFormat(self):
        length = 9
        data = tf.random.uniform((5, length, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "data_format"):
            conv_layers.separable_conv1d(data, 32, 3, data_format="invalid")

    def testInvalidStrides(self):
        length = 9
        data = tf.random.uniform((5, length, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.separable_conv1d(data, 32, 3, strides=(1, 2))

        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.separable_conv1d(data, 32, 3, strides=None)

    def testInvalidKernelSize(self):
        length = 9
        data = tf.random.uniform((5, length, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.separable_conv1d(data, 32, (1, 2))

        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.separable_conv1d(data, 32, None)

    def testCreateSeparableConv1D(self):
        length = 9
        data = tf.random.uniform((5, length, 4))
        layer = conv_layers.SeparableConv1D(32, 3, activation=tf.nn.relu)
        output = layer(data)
        if not tf.executing_eagerly():
            self.assertEqual(output.op.name, "separable_conv1d/Relu")
        self.assertEqual(output.get_shape().as_list(), [5, length - 2, 32])
        self.assertEqual(
            layer.depthwise_kernel.get_shape().as_list(), [3, 4, 1]
        )
        self.assertEqual(
            layer.pointwise_kernel.get_shape().as_list(), [1, 4, 32]
        )
        self.assertEqual(layer.bias.get_shape().as_list(), [32])

    def testCreateSeparableConv1DDepthMultiplier(self):
        length = 9
        data = tf.random.uniform((5, length, 4))
        layer = conv_layers.SeparableConv1D(32, 3, depth_multiplier=2)
        output = layer(data)
        self.assertEqual(output.get_shape().as_list(), [5, length - 2, 32])
        self.assertEqual(
            layer.depthwise_kernel.get_shape().as_list(), [3, 4, 2]
        )
        self.assertEqual(
            layer.pointwise_kernel.get_shape().as_list(), [1, 8, 32]
        )
        self.assertEqual(layer.bias.get_shape().as_list(), [32])

    def testCreateSeparableConv1DChannelsFirst(self):
        with tf.Graph().as_default():
            length = 9
            data = tf.random.uniform((5, 4, length))
            layer = conv_layers.SeparableConv1D(
                32, 3, data_format="channels_first"
            )
            output = layer(data)
            self.assertEqual(output.get_shape().as_list(), [5, 32, length - 2])
            self.assertEqual(
                layer.depthwise_kernel.get_shape().as_list(), [3, 4, 1]
            )
            self.assertEqual(
                layer.pointwise_kernel.get_shape().as_list(), [1, 4, 32]
            )
            self.assertEqual(layer.bias.get_shape().as_list(), [32])

    def testSeparableConv1DPaddingSame(self):
        length = 9
        data = tf.random.uniform((5, length, 32), seed=1)
        layer = conv_layers.SeparableConv1D(64, length, padding="same")
        output = layer(data)
        self.assertEqual(output.get_shape().as_list(), [5, length, 64])

    def testCreateSeparableConv1DWithStrides(self):
        length = 10
        data = tf.random.uniform((5, length, 3), seed=1)
        layer = conv_layers.SeparableConv1D(32, 3, strides=2, padding="same")
        output = layer(data)
        self.assertEqual(output.get_shape().as_list(), [5, length // 2, 32])

    def testCreateSeparableConv1DWithStridesChannelsFirst(self):
        with tf.Graph().as_default():
            data_format = "channels_first"
            length = 10
            data = tf.random.uniform((5, 3, length), seed=1)
            layer = conv_layers.SeparableConv1D(
                32, 3, strides=2, padding="same", data_format=data_format
            )
            output = layer(data)
            self.assertEqual(output.get_shape().as_list(), [5, 32, length // 2])

    def testFunctionalConv1DReuse(self):
        with tf.Graph().as_default():
            length = 10
            data = tf.random.uniform((5, length, 3), seed=1)
            conv_layers.separable_conv1d(data, 32, 3, name="sepconv1")
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)
            conv_layers.separable_conv1d(
                data, 32, 3, name="sepconv1", reuse=True
            )
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)

    def testFunctionalConv1DReuseFromScope(self):
        with tf.Graph().as_default():
            with tf.compat.v1.variable_scope("scope"):
                length = 10
                data = tf.random.uniform((5, length, 3), seed=1)
                conv_layers.separable_conv1d(data, 32, 3, name="sepconv1")
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)
            with tf.compat.v1.variable_scope("scope", reuse=True):
                conv_layers.separable_conv1d(data, 32, 3, name="sepconv1")
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)

    def testFunctionalConv1DNoReuse(self):
        with tf.Graph().as_default():
            length = 10
            data = tf.random.uniform((5, length, 3), seed=1)
            conv_layers.separable_conv1d(data, 32, 3)
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)
            conv_layers.separable_conv1d(data, 32, 3)
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 6)

    def testSeparableConv1DDepthwiseRegularizer(self):
        with tf.Graph().as_default():
            length = 9
            data = tf.random.uniform((5, length, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.SeparableConv1D(
                32, 3, depthwise_regularizer=reg
            )
            layer(data)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testSeparableConv1DPointwiseRegularizer(self):
        with tf.Graph().as_default():
            length = 9
            data = tf.random.uniform((5, length, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.SeparableConv1D(
                32, 3, pointwise_regularizer=reg
            )
            layer(data)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testSeparableConv1DBiasRegularizer(self):
        with tf.Graph().as_default():
            length = 9
            data = tf.random.uniform((5, length, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.SeparableConv1D(32, 3, bias_regularizer=reg)
            layer(data)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testSeparableConv1DNoBias(self):
        with tf.Graph().as_default():
            length = 9
            data = tf.random.uniform((5, length, 4))
            layer = conv_layers.SeparableConv1D(
                32, 3, activation=tf.nn.relu, use_bias=False
            )
            output = layer(data)
            self.assertEqual(output.op.name, "separable_conv1d/Relu")
            self.assertEqual(layer.bias, None)

    def testConstraints(self):
        d_constraint = lambda x: x / tf.reduce_sum(x)
        p_constraint = lambda x: x / tf.reduce_sum(x)
        b_constraint = lambda x: x / tf.reduce_max(x)
        layer = conv_layers.SeparableConv1D(
            2,
            3,
            depthwise_constraint=d_constraint,
            pointwise_constraint=p_constraint,
            bias_constraint=b_constraint,
        )
        inputs = tf.random.uniform((5, 3, 5), seed=1)
        layer(inputs)
        self.assertEqual(layer.depthwise_constraint, d_constraint)
        self.assertEqual(layer.pointwise_constraint, p_constraint)
        self.assertEqual(layer.bias_constraint, b_constraint)


class SeparableConv2DTest(tf.test.TestCase):
    def testInvalidDataFormat(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "data_format"):
            conv_layers.separable_conv2d(images, 32, 3, data_format="invalid")

    def testInvalidStrides(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.separable_conv2d(images, 32, 3, strides=(1, 2, 3))

        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.separable_conv2d(images, 32, 3, strides=None)

    def testInvalidKernelSize(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.separable_conv2d(images, 32, (1, 2, 3))

        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.separable_conv2d(images, 32, None)

    def testCreateSeparableConv2D(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4))
        layer = conv_layers.SeparableConv2D(32, [3, 3], activation=tf.nn.relu)
        output = layer(images)
        if not tf.executing_eagerly():
            self.assertEqual(output.op.name, "separable_conv2d/Relu")
        self.assertListEqual(
            output.get_shape().as_list(), [5, height - 2, width - 2, 32]
        )
        self.assertListEqual(
            layer.depthwise_kernel.get_shape().as_list(), [3, 3, 4, 1]
        )
        self.assertListEqual(
            layer.pointwise_kernel.get_shape().as_list(), [1, 1, 4, 32]
        )
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testCreateSeparableConv2DDepthMultiplier(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4))
        layer = conv_layers.SeparableConv2D(32, [3, 3], depth_multiplier=2)
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height - 2, width - 2, 32]
        )
        self.assertListEqual(
            layer.depthwise_kernel.get_shape().as_list(), [3, 3, 4, 2]
        )
        self.assertListEqual(
            layer.pointwise_kernel.get_shape().as_list(), [1, 1, 8, 32]
        )
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testCreateSeparableConv2DIntegerKernelSize(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4))
        layer = conv_layers.SeparableConv2D(32, 3)
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height - 2, width - 2, 32]
        )
        self.assertListEqual(
            layer.depthwise_kernel.get_shape().as_list(), [3, 3, 4, 1]
        )
        self.assertListEqual(
            layer.pointwise_kernel.get_shape().as_list(), [1, 1, 4, 32]
        )
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testCreateSeparableConv2DChannelsFirst(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, 4, height, width))
            layer = conv_layers.SeparableConv2D(
                32, [3, 3], data_format="channels_first"
            )
            output = layer(images)
            self.assertListEqual(
                output.get_shape().as_list(), [5, 32, height - 2, width - 2]
            )
            self.assertListEqual(
                layer.depthwise_kernel.get_shape().as_list(), [3, 3, 4, 1]
            )
            self.assertListEqual(
                layer.pointwise_kernel.get_shape().as_list(), [1, 1, 4, 32]
            )
            self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testSeparableConv2DPaddingSame(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 32), seed=1)
        layer = conv_layers.SeparableConv2D(
            64, images.get_shape()[1:3], padding="same"
        )
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height, width, 64]
        )

    def testCreateSeparableConvWithStrides(self):
        with tf.Graph().as_default():
            height, width = 6, 8
            # Test strides tuple
            images = tf.random.uniform((5, height, width, 3), seed=1)
            layer = conv_layers.SeparableConv2D(
                32, [3, 3], strides=(2, 2), padding="same"
            )
            output = layer(images)
            self.assertListEqual(
                output.get_shape().as_list(), [5, height / 2, width / 2, 32]
            )

            # Test strides integer
            layer = conv_layers.SeparableConv2D(
                32, [3, 3], strides=2, padding="same"
            )
            output = layer(images)
            self.assertListEqual(
                output.get_shape().as_list(), [5, height / 2, width / 2, 32]
            )

            # Test unequal strides
            layer = conv_layers.SeparableConv2D(
                32, [3, 3], strides=(2, 1), padding="same"
            )
            output = layer(images)
            self.assertListEqual(
                output.get_shape().as_list(), [5, height / 2, width, 32]
            )

    def testCreateSeparableConvWithStridesChannelsFirst(self):
        with tf.Graph().as_default():
            data_format = "channels_first"
            height, width = 6, 8
            # Test strides tuple
            images = tf.random.uniform((5, 3, height, width), seed=1)
            layer = conv_layers.SeparableConv2D(
                32,
                [3, 3],
                strides=(2, 2),
                padding="same",
                data_format=data_format,
            )
            output = layer(images)
            self.assertListEqual(
                output.get_shape().as_list(), [5, 32, height / 2, width / 2]
            )

            # Test strides integer
            layer = conv_layers.SeparableConv2D(
                32, [3, 3], strides=2, padding="same", data_format=data_format
            )
            output = layer(images)
            self.assertListEqual(
                output.get_shape().as_list(), [5, 32, height / 2, width / 2]
            )

            # Test unequal strides
            layer = conv_layers.SeparableConv2D(
                32,
                [3, 3],
                strides=(2, 1),
                padding="same",
                data_format=data_format,
            )
            output = layer(images)
            self.assertListEqual(
                output.get_shape().as_list(), [5, 32, height / 2, width]
            )

    def testFunctionalConv2DReuse(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 3), seed=1)
            conv_layers.separable_conv2d(images, 32, [3, 3], name="sepconv1")
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)
            conv_layers.separable_conv2d(
                images, 32, [3, 3], name="sepconv1", reuse=True
            )
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)

    def testFunctionalConv2DReuseFromScope(self):
        with tf.Graph().as_default():
            with tf.compat.v1.variable_scope("scope"):
                height, width = 7, 9
                images = tf.random.uniform((5, height, width, 3), seed=1)
                conv_layers.separable_conv2d(
                    images, 32, [3, 3], name="sepconv1"
                )
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)
            with tf.compat.v1.variable_scope("scope", reuse=True):
                conv_layers.separable_conv2d(
                    images, 32, [3, 3], name="sepconv1"
                )
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)

    def testFunctionalConv2DInitializerFromScope(self):
        with tf.Graph().as_default(), self.cached_session():
            with tf.compat.v1.variable_scope(
                "scope", initializer=tf.compat.v1.ones_initializer()
            ):
                height, width = 7, 9
                images = tf.random.uniform((5, height, width, 3), seed=1)
                conv_layers.separable_conv2d(
                    images, 32, [3, 3], name="sepconv1"
                )
                weights = tf.compat.v1.trainable_variables()
                # Check the names of weights in order.
                self.assertTrue("depthwise_kernel" in weights[0].name)
                self.assertTrue("pointwise_kernel" in weights[1].name)
                self.assertTrue("bias" in weights[2].name)
                self.evaluate(tf.compat.v1.global_variables_initializer())
                weights = self.evaluate(weights)
                # Check that the kernel weights got initialized to ones (from
                # scope)
                self.assertAllClose(weights[0], np.ones((3, 3, 3, 1)))
                self.assertAllClose(weights[1], np.ones((1, 1, 3, 32)))
                # Check that the bias still got initialized to zeros.
                self.assertAllClose(weights[2], np.zeros((32)))

    def testFunctionalConv2DNoReuse(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 3), seed=1)
            conv_layers.separable_conv2d(images, 32, [3, 3])
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 3)
            conv_layers.separable_conv2d(images, 32, [3, 3])
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 6)

    def testSeparableConv2DDepthwiseRegularizer(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.SeparableConv2D(
                32, [3, 3], depthwise_regularizer=reg
            )
            layer(images)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testSeparableConv2DPointwiseRegularizer(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.SeparableConv2D(
                32, [3, 3], pointwise_regularizer=reg
            )
            layer(images)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testSeparableConv2DBiasRegularizer(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.SeparableConv2D(
                32, [3, 3], bias_regularizer=reg
            )
            layer(images)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testSeparableConv2DNoBias(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 4))
            layer = conv_layers.SeparableConv2D(
                32, [3, 3], activation=tf.nn.relu, use_bias=False
            )
            output = layer(images)
            self.assertEqual(output.op.name, "separable_conv2d/Relu")
            self.assertListEqual(
                output.get_shape().as_list(), [5, height - 2, width - 2, 32]
            )
            self.assertListEqual(
                layer.depthwise_kernel.get_shape().as_list(), [3, 3, 4, 1]
            )
            self.assertListEqual(
                layer.pointwise_kernel.get_shape().as_list(), [1, 1, 4, 32]
            )
            self.assertEqual(layer.bias, None)

    def testConstraints(self):
        d_constraint = lambda x: x / tf.reduce_sum(x)
        p_constraint = lambda x: x / tf.reduce_sum(x)
        b_constraint = lambda x: x / tf.reduce_max(x)
        layer = conv_layers.SeparableConv2D(
            2,
            3,
            depthwise_constraint=d_constraint,
            pointwise_constraint=p_constraint,
            bias_constraint=b_constraint,
        )
        inputs = tf.random.uniform((5, 3, 3, 5), seed=1)
        layer(inputs)
        self.assertEqual(layer.depthwise_constraint, d_constraint)
        self.assertEqual(layer.pointwise_constraint, p_constraint)
        self.assertEqual(layer.bias_constraint, b_constraint)


class Conv2DTransposeTest(tf.test.TestCase):
    def testInvalidDataFormat(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "data_format"):
            conv_layers.conv2d_transpose(images, 32, 3, data_format="invalid")

    def testInvalidStrides(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.conv2d_transpose(images, 32, 3, strides=(1, 2, 3))

        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.conv2d_transpose(images, 32, 3, strides=None)

    def testInvalidKernelSize(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 3), seed=1)
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.conv2d_transpose(images, 32, (1, 2, 3))

        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.conv2d_transpose(images, 32, None)

    def testCreateConv2DTranspose(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4))
        layer = conv_layers.Conv2DTranspose(32, [3, 3], activation=tf.nn.relu)
        output = layer(images)
        if not tf.executing_eagerly():
            self.assertEqual(output.op.name, "conv2d_transpose/Relu")
        self.assertListEqual(
            output.get_shape().as_list(), [5, height + 2, width + 2, 32]
        )
        self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 32, 4])
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testConv2DTransposeFloat16(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4), dtype="float16")
        output = conv_layers.conv2d_transpose(
            images, 32, [3, 3], activation=tf.nn.relu
        )
        self.assertListEqual(
            output.get_shape().as_list(), [5, height + 2, width + 2, 32]
        )

    def testCreateConv2DTransposeIntegerKernelSize(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 4))
        layer = conv_layers.Conv2DTranspose(32, 3)
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height + 2, width + 2, 32]
        )
        self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 32, 4])
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testCreateConv2DTransposeChannelsFirst(self):
        height, width = 7, 9
        images = tf.random.uniform((5, 4, height, width))
        layer = conv_layers.Conv2DTranspose(
            32, [3, 3], data_format="channels_first"
        )
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, 32, height + 2, width + 2]
        )
        self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 32, 4])
        self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    def testConv2DTransposePaddingSame(self):
        height, width = 7, 9
        images = tf.random.uniform((5, height, width, 32), seed=1)
        layer = conv_layers.Conv2DTranspose(
            64, images.get_shape()[1:3], padding="same"
        )
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height, width, 64]
        )

    def testCreateConv2DTransposeWithStrides(self):
        height, width = 6, 8
        # Test strides tuple
        images = tf.random.uniform((5, height, width, 3), seed=1)
        layer = conv_layers.Conv2DTranspose(
            32, [3, 3], strides=(2, 2), padding="same"
        )
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height * 2, width * 2, 32]
        )

        # Test strides integer
        layer = conv_layers.Conv2DTranspose(
            32, [3, 3], strides=2, padding="same"
        )
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height * 2, width * 2, 32]
        )

        # Test unequal strides
        layer = conv_layers.Conv2DTranspose(
            32, [3, 3], strides=(2, 1), padding="same"
        )
        output = layer(images)
        self.assertListEqual(
            output.get_shape().as_list(), [5, height * 2, width, 32]
        )

    def testConv2DTransposeKernelRegularizer(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.Conv2DTranspose(
                32, [3, 3], kernel_regularizer=reg
            )
            layer(images)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testConv2DTransposeBiasRegularizer(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 4))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.Conv2DTranspose(
                32, [3, 3], bias_regularizer=reg
            )
            layer(images)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testConv2DTransposeNoBias(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 4))
            layer = conv_layers.Conv2DTranspose(
                32, [3, 3], activation=tf.nn.relu, use_bias=False
            )
            output = layer(images)
            self.assertEqual(output.op.name, "conv2d_transpose/Relu")
            self.assertListEqual(
                output.get_shape().as_list(), [5, height + 2, width + 2, 32]
            )
            self.assertListEqual(
                layer.kernel.get_shape().as_list(), [3, 3, 32, 4]
            )
            self.assertEqual(layer.bias, None)

    def testFunctionalConv2DTransposeReuse(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 3), seed=1)
            conv_layers.conv2d_transpose(images, 32, [3, 3], name="deconv1")
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)
            conv_layers.conv2d_transpose(
                images, 32, [3, 3], name="deconv1", reuse=True
            )
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)

    def testFunctionalConv2DTransposeReuseFromScope(self):
        with tf.Graph().as_default():
            with tf.compat.v1.variable_scope("scope"):
                height, width = 7, 9
                images = tf.random.uniform((5, height, width, 3), seed=1)
                conv_layers.conv2d_transpose(images, 32, [3, 3], name="deconv1")
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)
            with tf.compat.v1.variable_scope("scope", reuse=True):
                conv_layers.conv2d_transpose(images, 32, [3, 3], name="deconv1")
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)

    def testFunctionalConv2DTransposeInitializerFromScope(self):
        with tf.Graph().as_default(), self.cached_session():
            with tf.compat.v1.variable_scope(
                "scope", initializer=tf.compat.v1.ones_initializer()
            ):
                height, width = 7, 9
                images = tf.random.uniform((5, height, width, 3), seed=1)
                conv_layers.conv2d_transpose(images, 32, [3, 3], name="deconv1")
                weights = tf.compat.v1.trainable_variables()
                # Check the names of weights in order.
                self.assertTrue("kernel" in weights[0].name)
                self.assertTrue("bias" in weights[1].name)
                self.evaluate(tf.compat.v1.global_variables_initializer())
                weights = self.evaluate(weights)
                # Check that the kernel weights got initialized to ones (from
                # scope)
                self.assertAllClose(weights[0], np.ones((3, 3, 32, 3)))
                # Check that the bias still got initialized to zeros.
                self.assertAllClose(weights[1], np.zeros((32)))

    def testFunctionalConv2DTransposeNoReuse(self):
        with tf.Graph().as_default():
            height, width = 7, 9
            images = tf.random.uniform((5, height, width, 3), seed=1)
            conv_layers.conv2d_transpose(images, 32, [3, 3])
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)
            conv_layers.conv2d_transpose(images, 32, [3, 3])
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 4)

    def testConstraints(self):
        k_constraint = lambda x: x / tf.reduce_sum(x)
        b_constraint = lambda x: x / tf.reduce_max(x)
        layer = conv_layers.Conv2DTranspose(
            2, 3, kernel_constraint=k_constraint, bias_constraint=b_constraint
        )
        inputs = tf.random.uniform((5, 3, 3, 5), seed=1)
        layer(inputs)
        self.assertEqual(layer.kernel_constraint, k_constraint)
        self.assertEqual(layer.bias_constraint, b_constraint)


class Conv3DTransposeTest(tf.test.TestCase):
    def testInvalidDataFormat(self):
        depth, height, width = 5, 7, 9
        volumes = tf.random.uniform((5, depth, height, width, 32), seed=1)
        with self.assertRaisesRegex(ValueError, "data_format"):
            conv_layers.conv3d_transpose(volumes, 4, 3, data_format="invalid")

    def testInvalidStrides(self):
        depth, height, width = 5, 7, 9
        volumes = tf.random.uniform((5, depth, height, width, 32), seed=1)
        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.conv3d_transpose(volumes, 4, 3, strides=(1, 2))

        with self.assertRaisesRegex(ValueError, "strides"):
            conv_layers.conv3d_transpose(volumes, 4, 3, strides=None)

    def testInvalidKernelSize(self):
        depth, height, width = 5, 7, 9
        volumes = tf.random.uniform((5, depth, height, width, 32), seed=1)
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.conv3d_transpose(volumes, 4, (1, 2))

        with self.assertRaisesRegex(ValueError, "kernel_size"):
            conv_layers.conv3d_transpose(volumes, 4, None)

    def testCreateConv3DTranspose(self):
        depth, height, width = 5, 7, 9
        volumes = tf.random.uniform((5, depth, height, width, 32))
        layer = conv_layers.Conv3DTranspose(4, [3, 3, 3], activation=tf.nn.relu)
        output = layer(volumes)
        if not tf.executing_eagerly():
            self.assertEqual(output.op.name, "conv3d_transpose/Relu")
        self.assertListEqual(
            output.get_shape().as_list(),
            [5, depth + 2, height + 2, width + 2, 4],
        )
        self.assertListEqual(
            layer.kernel.get_shape().as_list(), [3, 3, 3, 4, 32]
        )
        self.assertListEqual(layer.bias.get_shape().as_list(), [4])

    def testCreateConv3DTransposeIntegerKernelSize(self):
        depth, height, width = 5, 7, 9
        volumes = tf.random.uniform((5, depth, height, width, 32))
        layer = conv_layers.Conv3DTranspose(4, 3)
        output = layer(volumes)
        self.assertListEqual(
            output.get_shape().as_list(),
            [5, depth + 2, height + 2, width + 2, 4],
        )
        self.assertListEqual(
            layer.kernel.get_shape().as_list(), [3, 3, 3, 4, 32]
        )
        self.assertListEqual(layer.bias.get_shape().as_list(), [4])

    def testCreateConv3DTransposeChannelsFirst(self):
        with tf.Graph().as_default():
            depth, height, width = 5, 7, 9
            volumes = tf.random.uniform((5, 32, depth, height, width))
            layer = conv_layers.Conv3DTranspose(
                4, [3, 3, 3], data_format="channels_first"
            )
            output = layer(volumes)
            self.assertListEqual(
                output.get_shape().as_list(),
                [5, 4, depth + 2, height + 2, width + 2],
            )
            self.assertListEqual(
                layer.kernel.get_shape().as_list(), [3, 3, 3, 4, 32]
            )
            self.assertListEqual(layer.bias.get_shape().as_list(), [4])

    def testConv3DTransposePaddingSame(self):
        depth, height, width = 5, 7, 9
        volumes = tf.random.uniform((5, depth, height, width, 64), seed=1)
        layer = conv_layers.Conv3DTranspose(
            32, volumes.get_shape()[1:4], padding="same"
        )
        output = layer(volumes)
        self.assertListEqual(
            output.get_shape().as_list(), [5, depth, height, width, 32]
        )

    def testCreateConv3DTransposeWithStrides(self):
        depth, height, width = 4, 6, 8
        # Test strides tuple.
        volumes = tf.random.uniform((5, depth, height, width, 32), seed=1)
        layer = conv_layers.Conv3DTranspose(
            4, [3, 3, 3], strides=(2, 2, 2), padding="same"
        )
        output = layer(volumes)
        self.assertListEqual(
            output.get_shape().as_list(),
            [5, depth * 2, height * 2, width * 2, 4],
        )

        # Test strides integer.
        layer = conv_layers.Conv3DTranspose(
            4, [3, 3, 3], strides=2, padding="same"
        )
        output = layer(volumes)
        self.assertListEqual(
            output.get_shape().as_list(),
            [5, depth * 2, height * 2, width * 2, 4],
        )

        # Test unequal strides.
        layer = conv_layers.Conv3DTranspose(
            4, [3, 3, 3], strides=(2, 1, 1), padding="same"
        )
        output = layer(volumes)
        self.assertListEqual(
            output.get_shape().as_list(), [5, depth * 2, height, width, 4]
        )

    def testConv3DTransposeKernelRegularizer(self):
        with tf.Graph().as_default():
            depth, height, width = 5, 7, 9
            volumes = tf.random.uniform((5, depth, height, width, 32))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.Conv3DTranspose(
                4, [3, 3, 3], kernel_regularizer=reg
            )
            layer(volumes)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testConv3DTransposeBiasRegularizer(self):
        with tf.Graph().as_default():
            depth, height, width = 5, 7, 9
            volumes = tf.random.uniform((5, depth, height, width, 32))
            reg = lambda x: 0.1 * tf.reduce_sum(x)
            layer = conv_layers.Conv3DTranspose(
                4, [3, 3, 3], bias_regularizer=reg
            )
            layer(volumes)
            loss_keys = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
            )
            self.assertEqual(len(loss_keys), 1)
            self.evaluate([v.initializer for v in layer.variables])
            self.assertListEqual(
                self.evaluate(layer.losses), self.evaluate(loss_keys)
            )

    def testConv3DTransposeNoBias(self):
        with tf.Graph().as_default():
            depth, height, width = 5, 7, 9
            volumes = tf.random.uniform((5, depth, height, width, 32))
            layer = conv_layers.Conv3DTranspose(
                4, [3, 3, 3], activation=tf.nn.relu, use_bias=False
            )
            output = layer(volumes)
            self.assertEqual(output.op.name, "conv3d_transpose/Relu")
            self.assertListEqual(
                output.get_shape().as_list(),
                [5, depth + 2, height + 2, width + 2, 4],
            )
            self.assertListEqual(
                layer.kernel.get_shape().as_list(), [3, 3, 3, 4, 32]
            )
            self.assertEqual(layer.bias, None)

    def testFunctionalConv3DTransposeReuse(self):
        with tf.Graph().as_default():
            depth, height, width = 5, 7, 9
            volumes = tf.random.uniform((5, depth, height, width, 32), seed=1)
            conv_layers.conv3d_transpose(volumes, 4, [3, 3, 3], name="deconv1")
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)
            conv_layers.conv3d_transpose(
                volumes, 4, [3, 3, 3], name="deconv1", reuse=True
            )
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)

    def testFunctionalConv3DTransposeReuseFromScope(self):
        with tf.Graph().as_default():
            with tf.compat.v1.variable_scope("scope"):
                depth, height, width = 5, 7, 9
                volumes = tf.random.uniform(
                    (5, depth, height, width, 32), seed=1
                )
                conv_layers.conv3d_transpose(
                    volumes, 4, [3, 3, 3], name="deconv1"
                )
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)
            with tf.compat.v1.variable_scope("scope", reuse=True):
                conv_layers.conv3d_transpose(
                    volumes, 4, [3, 3, 3], name="deconv1"
                )
                self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)

    def testFunctionalConv3DTransposeInitializerFromScope(self):
        with tf.Graph().as_default(), self.cached_session():
            with tf.compat.v1.variable_scope(
                "scope", initializer=tf.compat.v1.ones_initializer()
            ):
                depth, height, width = 5, 7, 9
                volumes = tf.random.uniform(
                    (5, depth, height, width, 32), seed=1
                )
                conv_layers.conv3d_transpose(
                    volumes, 4, [3, 3, 3], name="deconv1"
                )
                weights = tf.compat.v1.trainable_variables()
                # Check the names of weights in order.
                self.assertTrue("kernel" in weights[0].name)
                self.assertTrue("bias" in weights[1].name)
                self.evaluate(tf.compat.v1.global_variables_initializer())
                weights = self.evaluate(weights)
                # Check that the kernel weights got initialized to ones (from
                # scope)
                self.assertAllClose(weights[0], np.ones((3, 3, 3, 4, 32)))
                # Check that the bias still got initialized to zeros.
                self.assertAllClose(weights[1], np.zeros((4)))

    def testFunctionalConv3DTransposeNoReuse(self):
        with tf.Graph().as_default():
            depth, height, width = 5, 7, 9
            volumes = tf.random.uniform((5, depth, height, width, 32), seed=1)
            conv_layers.conv3d_transpose(volumes, 4, [3, 3, 3])
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 2)
            conv_layers.conv3d_transpose(volumes, 4, [3, 3, 3])
            self.assertEqual(len(tf.compat.v1.trainable_variables()), 4)

    def testConstraints(self):
        k_constraint = lambda x: x / tf.reduce_sum(x)
        b_constraint = lambda x: x / tf.reduce_max(x)
        layer = conv_layers.Conv3DTranspose(
            2, 3, kernel_constraint=k_constraint, bias_constraint=b_constraint
        )
        inputs = tf.random.uniform((5, 3, 3, 3, 5), seed=1)
        layer(inputs)
        self.assertEqual(layer.kernel_constraint, k_constraint)
        self.assertEqual(layer.bias_constraint, b_constraint)


if __name__ == "__main__":
    tf.test.main()
