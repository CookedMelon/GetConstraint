# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for object-based saving which use tf.train.* optimizers."""

import functools
import os

import tensorflow.compat.v2 as tf

from keras.engine import training
from keras.layers import core
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils

# isort: off
from tensorflow.python.checkpoint import (
    checkpoint as trackable_utils,
)
from tensorflow.python.eager import context
from tensorflow.python.framework import (
    test_util as tf_test_utils,
)


class NonLayerTrackable(tf.Module):
    def __init__(self):
        super().__init__()
        self.a_variable = trackable_utils.add_variable(
            self, name="a_variable", shape=[]
        )


class MyModel(training.Model):
    """A concrete Model for testing."""

    def __init__(self):
        super().__init__()
        self._named_dense = core.Dense(1, use_bias=True)
        self._second = core.Dense(1, use_bias=False)
        # We can still track Trackables which aren't Layers.
        self._non_layer = NonLayerTrackable()

    def call(self, values):
        ret = self._second(self._named_dense(values))
        return ret


class CheckpointingTests(test_combinations.TestCase):
    @tf_test_utils.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
    def testNamingWithOptimizer(self):
        input_value = tf.constant([[3.0]])
        model = MyModel()
        # A nuisance Model using the same optimizer. Its slot variables should
        # not go in the checkpoint, since it is never depended on.
        other_model = MyModel()
        optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
        optimizer_step = tf.compat.v1.train.get_or_create_global_step()
        root_trackable = tf.train.Checkpoint(
            optimizer=optimizer, model=model, optimizer_step=optimizer_step
        )
        if tf.executing_eagerly():
            optimizer.minimize(
                lambda: model(input_value), global_step=optimizer_step
            )
            optimizer.minimize(
                lambda: other_model(input_value), global_step=optimizer_step
            )
        else:
            train_op = optimizer.minimize(
                model(input_value), global_step=optimizer_step
            )
            optimizer.minimize(
                other_model(input_value), global_step=optimizer_step
            )
            self.evaluate(trackable_utils.gather_initializers(root_trackable))
            self.evaluate(train_op)
        (
            named_variables,
            serialized_graph,
            _,
        ) = tf.__internal__.tracking.ObjectGraphView(
            root_trackable
        ).serialize_object_graph()
        expected_checkpoint_names = (
            # Created in the root node, so no prefix.
            "optimizer_step",
            "model/_second/kernel",
            "model/_named_dense/kernel",
            "model/_named_dense/bias",
            # non-Layer dependency of the model
            "model/_non_layer/a_variable",
            # The optimizer creates two non-slot variables
            "optimizer/beta1_power",
            "optimizer/beta2_power",
            # Slot variables
            "model/_second/kernel/.OPTIMIZER_SLOT/optimizer/m",
            "model/_second/kernel/.OPTIMIZER_SLOT/optimizer/v",
            "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/m",
            "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/v",
            "model/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/m",
            "model/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/v",
        )
        suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
        expected_checkpoint_names = [
            name + suffix for name in expected_checkpoint_names
        ]
        named_variables = {v.name: v for v in named_variables}
        self.assertEqual(
            len(expected_checkpoint_names), len(named_variables.keys())
        )
        # Check that we've created the right full_names of objects (not
        # exhaustive)
        expected_names = {
            "optimizer_step" + suffix: "global_step",
            "model/_second/kernel" + suffix: "my_model/dense_1/kernel",
            "model/_named_dense/kernel" + suffix: "my_model/dense/kernel",
            "optimizer/beta1_power" + suffix: "beta1_power",
            "optimizer/beta2_power" + suffix: "beta2_power",
        }
        for nodes in serialized_graph.nodes:
            for attribute in nodes.attributes:
                expected_name = expected_names.pop(
                    attribute.checkpoint_key, None
                )
                if expected_name is not None:
                    self.assertEqual(expected_name, attribute.full_name)
        self.assertEmpty(expected_names)

        # Spot check the generated protocol buffers.
        self.assertEqual(
            "optimizer", serialized_graph.nodes[0].children[1].local_name
        )
        optimizer_node = serialized_graph.nodes[
            serialized_graph.nodes[0].children[1].node_id
        ]
        self.assertEqual("beta1_power", optimizer_node.children[0].local_name)
        self.assertEqual(
            "beta1_power",
            serialized_graph.nodes[optimizer_node.children[0].node_id]
            .attributes[0]
            .full_name,
        )
        self.assertEqual(
            "my_model/dense/kernel",
            serialized_graph.nodes[
                optimizer_node.slot_variables[0].original_variable_node_id
            ]
            .attributes[0]
            .full_name,
        )

        # We strip off the :0 suffix, as variable.name-based saving does.
        self.assertEqual(
            "my_model/dense/kernel/Adam",
            serialized_graph.nodes[
                optimizer_node.slot_variables[0].slot_variable_node_id
            ]
            .attributes[0]
            .full_name,
        )
        self.assertEqual(
            "my_model/dense/kernel/Adam:0",
            optimizer.get_slot(var=model._named_dense.kernel, name="m").name,
        )
        self.assertEqual(
            "model/_named_dense/kernel" + suffix,
            serialized_graph.nodes[
                optimizer_node.slot_variables[0].original_variable_node_id
            ]
            .attributes[0]
            .checkpoint_key,
        )
        self.assertEqual("m", optimizer_node.slot_variables[0].slot_name)
        self.assertEqual(
            "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/m" + suffix,
            serialized_graph.nodes[
                optimizer_node.slot_variables[0].slot_variable_node_id
            ]
            .attributes[0]
            .checkpoint_key,
        )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testSaveRestore(self):
        with self.test_session():
            model = MyModel()
            optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
            root_trackable = tf.train.Checkpoint(
                optimizer=optimizer, model=model
            )
            input_value = tf.constant([[3.0]])
            if tf.executing_eagerly():
                optimizer.minimize(lambda: model(input_value))
            else:
                train_op = optimizer.minimize(model(input_value))
                # TODO(allenl): Make initialization more pleasant when graph
                # building.
                root_trackable.save_counter
                self.evaluate(
                    trackable_utils.gather_initializers(root_trackable)
                )
                self.evaluate(train_op)
            prefix = os.path.join(self.get_temp_dir(), "ckpt")
            self.evaluate(
                tf.compat.v1.assign(model._named_dense.variables[1], [42.0])
            )
            m_bias_slot = optimizer.get_slot(
                model._named_dense.variables[1], "m"
            )
            self.evaluate(tf.compat.v1.assign(m_bias_slot, [1.5]))
            save_path = root_trackable.save(file_prefix=prefix)
            self.evaluate(
                tf.compat.v1.assign(model._named_dense.variables[1], [43.0])
            )
            self.evaluate(tf.compat.v1.assign(root_trackable.save_counter, 3))
            optimizer_variables = self.evaluate(optimizer.variables())
            self.evaluate(tf.compat.v1.assign(m_bias_slot, [-2.0]))
            # Immediate restoration
            status = root_trackable.restore(
                save_path=save_path
            ).assert_consumed()
            status.run_restore_ops()
            self.assertAllEqual(
                [42.0], self.evaluate(model._named_dense.variables[1])
            )
            self.assertAllEqual(1, self.evaluate(root_trackable.save_counter))
            self.assertAllEqual([1.5], self.evaluate(m_bias_slot))
            if not tf.executing_eagerly():
                # Restore-on-create is only supported when executing eagerly
                return
            on_create_model = MyModel()
            on_create_optimizer = tf.compat.v1.train.AdamOptimizer(
                0.001,
                # Preserve beta1_power and beta2_power when applying gradients
                # so we can test that they've been restored correctly.
                beta1=1.0,
                beta2=1.0,
            )
            on_create_root = tf.train.Checkpoint(
                optimizer=on_create_optimizer, model=on_create_model
            )
            # Deferred restoration
            status = on_create_root.restore(save_path=save_path)
            status.assert_nontrivial_match()
            status.assert_existing_objects_matched()
            with self.assertRaises(AssertionError):
                status.assert_consumed()
            on_create_model(tf.constant([[3.0]]))  # create variables
            self.assertAllEqual(1, self.evaluate(on_create_root.save_counter))
            self.assertAllEqual(
                [42.0], self.evaluate(on_create_model._named_dense.variables[1])
            )
            on_create_m_bias_slot = on_create_optimizer.get_slot(
                on_create_model._named_dense.variables[1], "m"
            )
            status.assert_existing_objects_matched()
            with self.assertRaises(AssertionError):
                status.assert_consumed()
            # Optimizer slot variables are created when the original variable is
            # restored.
            self.assertAllEqual([1.5], self.evaluate(on_create_m_bias_slot))
            self.assertAllEqual(
                optimizer_variables[2:],
                self.evaluate(on_create_optimizer.variables()),
            )
            dummy_var = tf.Variable([1.0])
            on_create_optimizer.minimize(loss=dummy_var.read_value)
            status.assert_existing_objects_matched()
            status.assert_consumed()
            (
                beta1_power,
                beta2_power,
            ) = on_create_optimizer._get_beta_accumulators()
            self.assertAllEqual(
                optimizer_variables[0], self.evaluate(beta1_power)
            )
            self.assertAllEqual(
                optimizer_variables[1], self.evaluate(beta2_power)
            )

    # TODO(allenl): Debug garbage created by this test in python3.
    def testDeferredRestorationUsageEager(self):
        """An idiomatic eager execution example."""
        num_training_steps = 10
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        for training_continuation in range(3):
            model = MyModel()
            optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
            root = tf.train.Checkpoint(
                optimizer=optimizer,
                model=model,
                optimizer_step=tf.compat.v1.train.get_or_create_global_step(),
            )
            root.restore(tf.train.latest_checkpoint(checkpoint_directory))
            for _ in range(num_training_steps):
                # TODO(allenl): Use a Dataset and serialize/checkpoint it.
                input_value = tf.constant([[3.0]])
                optimizer.minimize(
                    lambda: model(input_value),
                    global_step=root.optimizer_step,
                )
            root.save(file_prefix=checkpoint_prefix)
            self.assertEqual(
                (training_continuation + 1) * num_training_steps,
                root.optimizer_step.numpy(),
            )

    def testEagerDistributionStrategy(self):
        num_training_steps = 10
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

        def _train_fn(optimizer, model, root):
            input_value = tf.constant([[3.0]])
            optimizer.minimize(
                functools.partial(model, input_value),
                global_step=root.optimizer_step,
            )

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            for training_continuation in range(3):
                model = MyModel()
                optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
                root = tf.train.Checkpoint(
                    optimizer=optimizer,
                    model=model,
                    optimizer_step=tf.compat.v1.train.get_or_create_global_step(),  # noqa: E501
                )
                root.restore(tf.train.latest_checkpoint(checkpoint_directory))

                for _ in range(num_training_steps):
                    strategy.extended.call_for_each_replica(
                        functools.partial(_train_fn, optimizer, model, root)
                    )
                root.save(file_prefix=checkpoint_prefix)
                self.assertEqual(
                    (training_continuation + 1) * num_training_steps,
                    root.optimizer_step.numpy(),
                )

    def testGraphDistributionStrategy(self):
        self.skipTest("b/121381184")
        num_training_steps = 10
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

        def _train_fn(optimizer, model, root):
            input_value = tf.constant([[3.0]])
            return optimizer.minimize(
                functools.partial(model, input_value),
                global_step=root.optimizer_step,
            )

        for training_continuation in range(3):
            with tf.Graph().as_default():
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    model = MyModel()
                    optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
                    root = tf.train.Checkpoint(
                        optimizer=optimizer,
                        model=model,
                        optimizer_step=tf.compat.v1.train.get_or_create_global_step(),  # noqa: E501
                    )
                    status = root.restore(
                        tf.train.latest_checkpoint(checkpoint_directory)
                    )
                    train_op = strategy.extended.call_for_each_replica(
                        functools.partial(_train_fn, optimizer, model, root)
                    )
                    with self.session() as session:
                        if training_continuation > 0:
                            status.assert_consumed()
                        status.initialize_or_restore()
                        for _ in range(num_training_steps):
                            session.run(train_op)
                        root.save(file_prefix=checkpoint_prefix)
                self.assertEqual(
                    (training_continuation + 1) * num_training_steps,
                    root.optimizer_step.numpy(),
                )

    def testUsageGraph(self):
        """Expected usage when graph building."""
        with context.graph_mode():
            num_training_steps = 10
            checkpoint_directory = self.get_temp_dir()
            checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
            for training_continuation in range(3):
                with tf.Graph().as_default():
                    model = MyModel()
                    optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
                    root = tf.compat.v1.train.Checkpoint(
                        optimizer=optimizer,
                        model=model,
                        global_step=tf.compat.v1.train.get_or_create_global_step(),  # noqa: E501
                    )
                    input_value = tf.constant([[3.0]])
                    train_op = optimizer.minimize(
                        model(input_value), global_step=root.global_step
                    )
                    checkpoint_path = tf.train.latest_checkpoint(
                        checkpoint_directory
                    )
                    with self.session(
                        graph=tf.compat.v1.get_default_graph()
                    ) as session:
                        status = root.restore(save_path=checkpoint_path)
                        status.initialize_or_restore(session=session)
                        if checkpoint_path is None:
                            self.assertEqual(0, training_continuation)
                            with self.assertRaises(AssertionError):
                                status.assert_consumed()
                            with self.assertRaises(AssertionError):
                                status.assert_existing_objects_matched()
                        else:
                            status.assert_consumed()
                            status.assert_existing_objects_matched()
                        for _ in range(num_training_steps):
                            session.run(train_op)
                        root.save(
                            file_prefix=checkpoint_prefix, session=session
                        )
                        self.assertEqual(
                            (training_continuation + 1) * num_training_steps,
                            session.run(root.global_step),
                        )
                        self.assertEqual(
                            training_continuation + 1,
                            session.run(root.save_counter),
                        )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testAgnosticUsage(self):
        """Graph/eager agnostic usage."""
        # Does create garbage when executing eagerly due to ops.Graph()
        # creation.
        with self.test_session():
            num_training_steps = 10
            checkpoint_directory = self.get_temp_dir()
            for training_continuation in range(3):
                with test_utils.device(should_use_gpu=True):
                    model = MyModel()
                    optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
                    root = tf.train.Checkpoint(
                        optimizer=optimizer,
                        model=model,
                        global_step=tf.compat.v1.train.get_or_create_global_step(),  # noqa: E501
                    )
                    manager = tf.train.CheckpointManager(
                        root, checkpoint_directory, max_to_keep=1
                    )
                    status = root.restore(save_path=manager.latest_checkpoint)
                    input_value = tf.constant([[3.0]])
                    train_fn = functools.partial(
                        optimizer.minimize,
                        functools.partial(model, input_value),
                        global_step=root.global_step,
                    )
                    if not tf.executing_eagerly():
                        train_fn = functools.partial(self.evaluate, train_fn())
                    status.initialize_or_restore()
                    for _ in range(num_training_steps):
                        train_fn()
                    manager.save()
                    self.assertEqual(
                        (training_continuation + 1) * num_training_steps,
                        self.evaluate(root.global_step),
                    )
                    self.assertEqual(
                        training_continuation + 1,
                        self.evaluate(root.save_counter),
                    )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testWithDefun(self):
        with self.test_session():
            num_training_steps = 2
            checkpoint_directory = self.get_temp_dir()
            checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
            for training_continuation in range(3):
                with test_utils.device(should_use_gpu=True):
                    model = MyModel()
                    # Don't actually train so we can test variable values
                    optimizer = tf.compat.v1.train.AdamOptimizer(0.0)
                    root = tf.train.Checkpoint(
                        optimizer=optimizer,
                        model=model,
                        global_step=tf.compat.v1.train.get_or_create_global_step(),  # noqa: E501
                    )
                    checkpoint_path = tf.train.latest_checkpoint(
                        checkpoint_directory
                    )
                    status = root.restore(save_path=checkpoint_path)

                    def train_fn():
                        @tf.function
                        def _call_model(x):
                            return model(x)

                        with tf.GradientTape() as tape:
                            loss = _call_model(tf.constant([[3.0]]))
                        gradients = tape.gradient(loss, model.variables)
                        return optimizer.apply_gradients(
                            zip(gradients, model.variables),
                            global_step=root.global_step,
                        )

                    if not tf.executing_eagerly():
                        train_fn = functools.partial(self.evaluate, train_fn())
                    status.initialize_or_restore()
                    for _ in range(num_training_steps):
                        train_fn()
                    if training_continuation > 0:
                        status.assert_consumed()
                        self.assertAllClose(
                            [[42.0]], self.evaluate(model.variables[0])
                        )
                    else:
                        self.evaluate(model.variables[0].assign([[42.0]]))
                    root.save(file_prefix=checkpoint_prefix)
                    self.assertEqual(
                        (training_continuation + 1) * num_training_steps,
                        self.evaluate(root.global_step),
                    )
                    self.assertEqual(
                        training_continuation + 1,
                        self.evaluate(root.save_counter),
                    )

    @test_combinations.generate(test_combinations.combine(mode=["eager"]))
    def testAnonymousVarsInInit(self):
        class Model(training.Model):
            def __init__(self):
                super().__init__()
                self.w = tf.Variable(0.0)
                self.b = tf.Variable(0.0)
                self.vars = [self.w, self.b]

            def call(self, x):
                return x * self.w + self.b

        model = Model()
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.05)
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        for _ in range(2):
            checkpoint.save(checkpoint_prefix)
            with tf.GradientTape() as tape:
                loss = (tf.constant(1.0) - model(tf.constant(1.0))) ** 2
            grad = tape.gradient(loss, model.vars)
            optimizer.apply_gradients(
                [(g, v) for g, v in zip(grad, model.vars)]
            )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_initialize_if_not_restoring(self):
        with self.test_session():
            checkpoint_directory = self.get_temp_dir()
            checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
            optimizer_only_prefix = os.path.join(checkpoint_directory, "opt")
            with test_utils.device(should_use_gpu=True):
                model = MyModel()
                optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
                root = tf.train.Checkpoint(
                    # Do not save the optimizer with the checkpoint.
                    model=model,
                    global_step=tf.compat.v1.train.get_or_create_global_step(),
                )
                optimizer_checkpoint = tf.train.Checkpoint(optimizer=optimizer)

                checkpoint_path = tf.train.latest_checkpoint(
                    checkpoint_directory
                )
                status = root.restore(save_path=checkpoint_path)
                input_value = tf.constant([[3.0]])
                train_fn = functools.partial(
                    optimizer.minimize,
                    functools.partial(model, input_value),
                    global_step=root.global_step,
                )
                if not tf.executing_eagerly():
                    train_fn = functools.partial(self.evaluate, train_fn())
                status.initialize_or_restore()
                self.evaluate([v.initializer for v in optimizer.variables()])
                train_fn()
                model_save_path = root.save(file_prefix=checkpoint_prefix)
                self.evaluate(optimizer.variables()[0].assign(42.0))
                optimizer_save_path = optimizer_checkpoint.save(
                    optimizer_only_prefix
                )

            # Restore into a graph with the optimizer
            with test_utils.device(should_use_gpu=True):
                model = MyModel()
                optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
                root = tf.train.Checkpoint(
                    optimizer=optimizer,
                    model=model,
                    global_step=tf.compat.v1.train.get_or_create_global_step(),
                )
                status = root.restore(save_path=model_save_path)
                input_value = tf.constant([[3.0]])
                train_fn = functools.partial(
                    optimizer.minimize,
                    functools.partial(model, input_value),
                    global_step=root.global_step,
                )
                if not tf.executing_eagerly():
                    train_fn = functools.partial(self.evaluate, train_fn())
                status.initialize_or_restore()
                train_fn()
                with self.assertRaises(AssertionError):
                    status.assert_existing_objects_matched()
                with self.assertRaises(AssertionError):
                    status.assert_consumed()

            # Make sure initialization doesn't clobber later restores
            with test_utils.device(should_use_gpu=True):
                model = MyModel()
                optimizer = tf.compat.v1.train.AdamOptimizer(0.001, beta1=1.0)
                root = tf.train.Checkpoint(
                    optimizer=optimizer,
                    model=model,
                    global_step=tf.compat.v1.train.get_or_create_global_step(),
                )
                opt_root = tf.train.Checkpoint(optimizer=optimizer)
                status = root.restore(save_path=model_save_path)
                init_only_optimizer_status = opt_root.restore(save_path=None)
                optimizer_status = opt_root.restore(
                    save_path=optimizer_save_path
                )
                input_value = tf.constant([[3.0]])
                train_fn = functools.partial(
                    optimizer.minimize,
                    functools.partial(model, input_value),
                    global_step=root.global_step,
                )
                if not tf.executing_eagerly():
                    train_fn = functools.partial(self.evaluate, train_fn())
                optimizer_status.run_restore_ops()
                status.initialize_or_restore()
                init_only_optimizer_status.initialize_or_restore()
                train_fn()
                self.assertEqual(42.0, self.evaluate(optimizer.variables()[0]))


class CheckpointCompatibilityTests(test_combinations.TestCase):
    def _initialized_model(self):
        input_value = tf.constant([[3.0]])
        model = MyModel()
        optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
        optimizer_step = tf.compat.v1.train.get_or_create_global_step()
        root_trackable = tf.train.Checkpoint(
            optimizer=optimizer, model=model, optimizer_step=optimizer_step
        )
        train_op = optimizer.minimize(
            functools.partial(model, input_value), global_step=optimizer_step
        )
        self.evaluate(trackable_utils.gather_initializers(root_trackable))
        self.evaluate(train_op)
        # A regular variable, a slot variable, and a non-slot Optimizer variable
        # with known values to check when loading.
        self.evaluate(model._named_dense.bias.assign([1.0]))
        self.evaluate(
            optimizer.get_slot(var=model._named_dense.bias, name="m").assign(
                [2.0]
            )
        )
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.evaluate(beta1_power.assign(3.0))
        return root_trackable

    def _set_sentinels(self, root_trackable):
        self.evaluate(root_trackable.model._named_dense.bias.assign([101.0]))
        self.evaluate(
            root_trackable.optimizer.get_slot(
                var=root_trackable.model._named_dense.bias, name="m"
            ).assign([102.0])
        )
        beta1_power, _ = root_trackable.optimizer._get_beta_accumulators()
        self.evaluate(beta1_power.assign(103.0))

    def _check_sentinels(self, root_trackable):
        self.assertAllEqual(
            [1.0], self.evaluate(root_trackable.model._named_dense.bias)
        )
        self.assertAllEqual(
            [2.0],
            self.evaluate(
                root_trackable.optimizer.get_slot(
                    var=root_trackable.model._named_dense.bias, name="m"
                )
            ),
        )
        beta1_power, _ = root_trackable.optimizer._get_beta_accumulators()
        self.assertAllEqual(3.0, self.evaluate(beta1_power))

    def _write_name_based_checkpoint(self):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        with context.graph_mode():
            save_graph = tf.Graph()
            with save_graph.as_default(), self.session(
                graph=save_graph
            ) as session:
                root = self._initialized_model()
                name_saver = tf.compat.v1.train.Saver()
                return name_saver.save(
                    sess=session,
                    save_path=checkpoint_prefix,
                    global_step=root.optimizer_step,
                )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testLoadFromNameBasedSaver(self):
        """Save a name-based checkpoint, load it using the object-based API."""
        with test_utils.device(should_use_gpu=True):
            with self.test_session():
                save_path = self._write_name_based_checkpoint()
                root = self._initialized_model()
                self._set_sentinels(root)
                with self.assertRaises(AssertionError):
                    self._check_sentinels(root)
                object_saver = tf.train.Checkpoint(root=root)
                self._set_sentinels(root)
                status = object_saver.read(save_path)
                if tf.executing_eagerly():
                    self._check_sentinels(root)
                if tf.executing_eagerly():
                    status.assert_consumed()
                    status.assert_existing_objects_matched()
                    status.assert_nontrivial_match()
                else:
                    # When graph building, we haven't read any keys, so we don't
                    # know whether the restore will be complete.
                    with self.assertRaisesRegex(AssertionError, "not restored"):
                        status.assert_consumed()
                    with self.assertRaisesRegex(AssertionError, "not restored"):
                        status.assert_existing_objects_matched()
                    with self.assertRaisesRegex(AssertionError, "not restored"):
                        status.assert_nontrivial_match()
                status.run_restore_ops()
                self._check_sentinels(root)
                self._set_sentinels(root)
                status = object_saver.read(save_path)
                status.initialize_or_restore()
                self._check_sentinels(root)
                # Check that there is no error when keys are missing from the
                # name-based checkpoint.
                root.not_in_name_checkpoint = tf.Variable([1.0])
                status = object_saver.read(save_path)
                with self.assertRaises(AssertionError):
                    status.assert_existing_objects_matched()

    def testSaveGraphLoadEager(self):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        with context.graph_mode():
            save_graph = tf.Graph()
            with save_graph.as_default(), self.session(graph=save_graph):
                root = self._initialized_model()
                save_path = root.save(file_prefix=checkpoint_prefix)
        with tf.__internal__.eager_context.eager_mode():
            root = self._initialized_model()
            self._set_sentinels(root)
            root.restore(save_path).assert_consumed()
            self._check_sentinels(root)

    def testSaveEagerLoadGraph(self):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        with tf.__internal__.eager_context.eager_mode():
            root = self._initialized_model()
            save_path = root.save(file_prefix=checkpoint_prefix)
        with context.graph_mode():
            save_graph = tf.Graph()
            with save_graph.as_default(), self.session(graph=save_graph):
                root = self._initialized_model()
                self._set_sentinels(root)
                root.restore(save_path).assert_consumed().run_restore_ops()
                self._check_sentinels(root)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.test.main()
