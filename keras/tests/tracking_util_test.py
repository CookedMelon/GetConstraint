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

import functools
import os
import weakref

import tensorflow.compat.v2 as tf

from keras.engine import input_layer
from keras.engine import sequential
from keras.engine import training
from keras.layers import core
from keras.layers import reshaping
from keras.optimizers.legacy import adam
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
from tensorflow.python.platform import tf_logging as logging


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


class NonLayerTrackable(tf.Module):
    def __init__(self):
        super().__init__()
        self.a_variable = trackable_utils.add_variable(
            self, name="a_variable", shape=[]
        )


class InterfaceTests(tf.test.TestCase):
    def testLayerDeduplication(self):
        model = training.Model()
        layer_one = core.Dense(1)
        layer_two = core.Dense(1)
        model.other_path = [layer_one, layer_two]
        model.l2 = layer_two
        model.l1 = layer_one
        self.assertEqual([layer_one, layer_two], model.layers)

    def testSaveWithOnlyKerasSession(self):

        with tf.Graph().as_default(), self.cached_session():
            inp = input_layer.Input([1])
            dense = core.Dense(1)(inp)
            model = training.Model(inp, dense)
            model.compile(optimizer="sgd", loss="mse")
            model.fit([1.0], [2.0])
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))


class CheckpointingTests(test_combinations.TestCase):
    @tf_test_utils.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
    def testNamingWithOptimizer(self):
        input_value = tf.constant([[3.0]])
        model = MyModel()
        # A nuisance Model using the same optimizer. Its slot variables should
        # not go in the checkpoint, since it is never depended on.
        other_model = MyModel()
        optimizer = adam.Adam(0.001)
        step = tf.compat.v1.train.get_or_create_global_step()
        root_trackable = tf.train.Checkpoint(
            optimizer=optimizer, model=model, step=step
        )

        with tf.GradientTape() as tape:
            loss = model(input_value)
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        train_op = tf.group(
            optimizer.apply_gradients(zip(gradients, variables)),
            step.assign_add(1),
        )

        with tf.GradientTape() as tape:
            loss = other_model(input_value)
        variables = other_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        self.evaluate(trackable_utils.gather_initializers(root_trackable))
        self.evaluate(train_op)
        (
            named_variables,
            serialized_graph,
            _,
        ) = tf.__internal__.tracking.ObjectGraphView(
            root_trackable
        ).serialize_object_graph()
        expected_slot_keys = (
            "model/_second/kernel/.OPTIMIZER_SLOT/optimizer/m",
            "model/_second/kernel/.OPTIMIZER_SLOT/optimizer/v",
            "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/m",
            "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/v",
            "model/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/m",
            "model/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/v",
        )
        expected_checkpoint_names = (
            # Created in the root node, so no prefix.
            "step",
            "model/_second/kernel",
            "model/_named_dense/kernel",
            "model/_named_dense/bias",
            # non-Layer dependency of the model
            "model/_non_layer/a_variable",
            "optimizer/learning_rate",
            "optimizer/beta_1",
            "optimizer/beta_2",
            "optimizer/iter",
            "optimizer/decay",
        ) + expected_slot_keys
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
            "step" + suffix: "global_step",
            "model/_second/kernel" + suffix: "my_model/dense_1/kernel",
            "model/_named_dense/kernel" + suffix: "my_model/dense/kernel",
            "optimizer/beta_1" + suffix: "Adam/beta_1",
            "optimizer/beta_2" + suffix: "Adam/beta_2",
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
        children = [node.local_name for node in optimizer_node.children]
        self.assertEqual(
            # hyper variable dependencies
            len(["beta_1", "beta_2", "iter", "decay", "learning_rate"]),
            len(children),
        )
        serialized_slot_keys = []
        for slot in optimizer_node.slot_variables:
            for attribute in serialized_graph.nodes[
                slot.slot_variable_node_id
            ].attributes:
                serialized_slot_keys.append(attribute.checkpoint_key)
        self.assertEqual(
            len([key + suffix for key in expected_slot_keys]),
            len(serialized_slot_keys),
        )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testSaveRestore(self):
        with self.test_session():
            model = MyModel()
            optimizer = adam.Adam(0.001)
            root_trackable = tf.train.Checkpoint(
                optimizer=optimizer, model=model
            )
            input_value = tf.constant([[3.0]])
            with tf.GradientTape() as tape:
                loss = model(input_value)
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
            self.assertFalse(root_trackable.save_counter.trainable)
            self.evaluate(trackable_utils.gather_initializers(root_trackable))
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
            optimizer_variables = self.evaluate(
                sorted(optimizer.variables(), key=lambda v: v.name)
            )
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
            on_create_optimizer = adam.Adam(0.001)
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
            if not tf.executing_eagerly():
                with self.assertRaises(AssertionError):
                    status.assert_consumed()
            # Optimizer slot variables are created when the original variable is
            # restored.
            self.assertAllEqual([1.5], self.evaluate(on_create_m_bias_slot))
            dummy_var = tf.Variable([1.0])
            on_create_optimizer.minimize(
                loss=dummy_var.read_value, var_list=[dummy_var]
            )
            status.assert_existing_objects_matched()
            status.assert_consumed()
            self.assertAllEqual(
                optimizer_variables,
                # Creation order is different, so .variables() needs to be
                # re-sorted.
                self.evaluate(
                    sorted(optimizer.variables(), key=lambda v: v.name)
                ),
            )

    # TODO(allenl): Debug garbage created by this test in python3.
    def testDeferredRestorationUsageEager(self):
        """An idiomatic eager execution example."""
        num_training_steps = 10
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        for training_continuation in range(3):
            model = MyModel()
            optimizer = adam.Adam(0.001)
            root = tf.train.Checkpoint(optimizer=optimizer, model=model)
            root.restore(tf.train.latest_checkpoint(checkpoint_directory))
            for _ in range(num_training_steps):
                # TODO(allenl): Use a Dataset and serialize/checkpoint it.
                input_value = tf.constant([[3.0]])
                with tf.GradientTape() as tape:
                    loss = model(input_value)
                variables = model.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
            root.save(file_prefix=checkpoint_prefix)
            self.assertEqual(
                (training_continuation + 1) * num_training_steps,
                root.optimizer.iterations.numpy(),
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
                    optimizer = adam.Adam(0.001)
                    root = tf.compat.v1.train.Checkpoint(
                        optimizer=optimizer, model=model
                    )
                    input_value = tf.constant([[3.0]])
                    with tf.GradientTape() as tape:
                        loss = model(input_value)
                    variables = model.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    train_op = optimizer.apply_gradients(
                        zip(gradients, variables)
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
                            session.run(root.optimizer.iterations),
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
            optimizer = adam.Adam(0.001)

            def _train_fn(model, input_value):
                with tf.GradientTape() as tape:
                    loss = model(input_value)
                variables = model.trainable_variables
                gradients = tape.gradient(loss, variables)
                return optimizer.apply_gradients(zip(gradients, variables))

            for training_continuation in range(3):
                with test_utils.device(should_use_gpu=True):
                    model = MyModel()
                    root = tf.train.Checkpoint(optimizer=optimizer, model=model)
                    manager = tf.train.CheckpointManager(
                        root, checkpoint_directory, max_to_keep=1
                    )
                    status = root.restore(save_path=manager.latest_checkpoint)
                    input_value = tf.constant([[3.0]])
                    train_fn = functools.partial(_train_fn, model, input_value)
                    if not tf.executing_eagerly():
                        train_fn = functools.partial(self.evaluate, train_fn())
                    status.initialize_or_restore()
                    for _ in range(num_training_steps):
                        train_fn()
                    manager.save()
                    self.assertEqual(
                        (training_continuation + 1) * num_training_steps,
                        self.evaluate(root.optimizer.iterations),
                    )
                    self.assertEqual(
                        training_continuation + 1,
                        self.evaluate(root.save_counter),
                    )

    @test_combinations.generate(test_combinations.combine(mode=["eager"]))
    def testPartialRestoreWarningObject(self):
        optimizer = adam.Adam(0.0)
        original_root = tf.train.Checkpoint(
            v1=tf.Variable(2.0), v2=tf.Variable(3.0), optimizer=optimizer
        )
        # Create a slot variable to save
        optimizer.minimize(original_root.v1.read_value, [original_root.v1])
        prefix = os.path.join(self.get_temp_dir(), "ckpt")
        save_path = original_root.save(prefix)
        partial_root = tf.train.Checkpoint(v1=tf.Variable(0.0))
        weak_partial_root = weakref.ref(partial_root)
        weak_v1 = weakref.ref(partial_root.v1)
        partial_root.restore(save_path)
        self.assertEqual(2.0, partial_root.v1.numpy())
        with tf.compat.v1.test.mock.patch.object(
            logging, "warning"
        ) as mock_log:
            del partial_root
            self.assertIsNone(weak_partial_root())
            self.assertIsNone(weak_v1())
            messages = str(mock_log.call_args_list)
        self.assertIn("(root).v2'", messages)
        self.assertIn("(root).optimizer's state 'm' for (root).v1", messages)
        self.assertNotIn("(root).v1'", messages)
        self.assertIn("expect_partial()", messages)

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
                    optimizer = adam.Adam(0.0)
                    root = tf.train.Checkpoint(optimizer=optimizer, model=model)
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
                            zip(gradients, model.variables)
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
                        self.evaluate(optimizer.iterations),
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
        optimizer = adam.Adam(learning_rate=0.05)
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
    def testDeferredSlotRestoration(self):
        with self.test_session():
            checkpoint_directory = self.get_temp_dir()

            root = tf.train.Checkpoint()
            root.var = trackable_utils.add_variable(
                root, name="var", initializer=0.0
            )
            optimizer = adam.Adam(0.1)
            variables = [root.var]
            gradients = [1.0]
            train_op = optimizer.apply_gradients(zip(gradients, variables))
            # Note that `optimizer` has not been added as a dependency of
            # `root`. Create a one-off grouping so that slot variables for
            # `root.var` get initialized too.
            self.evaluate(
                trackable_utils.gather_initializers(
                    tf.train.Checkpoint(root=root, optimizer=optimizer)
                )
            )
            self.evaluate(train_op)
            self.evaluate(tf.compat.v1.assign(root.var, 12.0))
            no_slots_path = root.save(
                os.path.join(checkpoint_directory, "no_slots")
            )
            root.optimizer = optimizer
            self.evaluate(tf.compat.v1.assign(root.var, 13.0))
            self.evaluate(
                tf.compat.v1.assign(
                    optimizer.get_slot(slot_name="m", var=root.var), 14.0
                )
            )
            slots_path = root.save(
                os.path.join(checkpoint_directory, "with_slots")
            )
            new_root = tf.train.Checkpoint()
            # Load the slot-containing checkpoint (deferred), then immediately
            # overwrite the non-slot variable (also deferred).
            slot_status = new_root.restore(slots_path)
            no_slot_status = new_root.restore(no_slots_path)
            with self.assertRaises(AssertionError):
                no_slot_status.assert_consumed()
            new_root.var = trackable_utils.add_variable(
                new_root, name="var", shape=[]
            )
            no_slot_status.assert_consumed()
            no_slot_status.run_restore_ops()
            self.assertEqual(12.0, self.evaluate(new_root.var))
            new_root.optimizer = adam.Adam(0.1)
            slot_status.assert_existing_objects_matched()
            if not tf.executing_eagerly():
                with self.assertRaisesRegex(
                    AssertionError, "Unresolved object"
                ):
                    slot_status.assert_consumed()
            self.assertEqual(12.0, self.evaluate(new_root.var))
            if tf.executing_eagerly():
                # Slot variables are only created with restoring initializers
                # when executing eagerly.
                self.assertEqual(
                    14.0,
                    self.evaluate(
                        new_root.optimizer.get_slot(
                            slot_name="m", var=new_root.var
                        )
                    ),
                )
            else:
                # Slot variables are not created eagerly when graph building.
                with self.assertRaises(KeyError):
                    new_root.optimizer.get_slot(slot_name="m", var=new_root.var)
            variables = [new_root.var]
            gradients = [1.0]
            train_op = new_root.optimizer.apply_gradients(
                zip(gradients, variables)
            )
            # The slot variable now exists; restore() didn't create it, but we
            # should now have a restore op for it.
            slot_status.run_restore_ops()
            if not tf.executing_eagerly():
                # The train op hasn't run when graph building, so the slot
                # variable has its restored value. It has run in eager, so the
                # value will be different.
                self.assertEqual(
                    14.0,
                    self.evaluate(
                        new_root.optimizer.get_slot(
                            slot_name="m", var=new_root.var
                        )
                    ),
                )
            self.evaluate(train_op)
            slot_status.assert_consumed()

    def testManySavesGraph(self):
        """Saves after the first should not modify the graph."""
        with context.graph_mode():
            graph = tf.Graph()
            with graph.as_default(), self.session(graph):
                checkpoint_directory = self.get_temp_dir()
                checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
                obj = tf.train.Checkpoint()
                obj.var = tf.Variable(0.0, name="v")
                obj.opt = adam.Adam(0.1)
                variables = [obj.var]
                gradients = [1.0]
                obj.opt.apply_gradients(zip(gradients, variables))
                self.evaluate(trackable_utils.gather_initializers(obj))
                obj.save(checkpoint_prefix)
                graph.finalize()
                obj.save(checkpoint_prefix)

    def testManyRestoresGraph(self):
        """Restores after the first should not modify the graph."""
        with context.graph_mode():
            graph = tf.Graph()
            with graph.as_default(), self.session(graph):
                checkpoint_directory = self.get_temp_dir()
                checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
                obj = tf.train.Checkpoint()
                obj.var = tf.Variable(0.0, name="v")
                obj.opt = adam.Adam(0.1)
                variables = [obj.var]
                gradients = [1.0]
                obj.opt.apply_gradients(zip(gradients, variables))
                self.evaluate(trackable_utils.gather_initializers(obj))
                save_path = obj.save(checkpoint_prefix)
                obj.restore(save_path)
                graph.finalize()
                obj.restore(save_path)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_sequential(self):
        with self.test_session():
            model = sequential.Sequential()
            checkpoint = tf.train.Checkpoint(model=model)
            model.add(core.Dense(4))
            second_dense = core.Dense(5)
            model.add(second_dense)
            model(tf.constant([[1.0]]))
            checkpoint.restore(None).initialize_or_restore()
            self.evaluate(
                second_dense.bias.assign(tf.constant([1.0, 2.0, 3.0, 4.0, 5.0]))
            )
            checkpoint_directory = self.get_temp_dir()
            checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
            save_path = checkpoint.save(checkpoint_prefix)
            self.evaluate(
                second_dense.bias.assign(tf.constant([5.0, 6.0, 7.0, 8.0, 9.0]))
            )
            checkpoint.restore(save_path).assert_consumed().run_restore_ops()
            self.assertAllEqual(
                [1.0, 2.0, 3.0, 4.0, 5.0], self.evaluate(second_dense.bias)
            )

            deferred_sequential = sequential.Sequential()
            deferred_sequential_checkpoint = tf.train.Checkpoint(
                model=deferred_sequential
            )
            status = deferred_sequential_checkpoint.restore(save_path)
            deferred_sequential.add(core.Dense(4))
            deferred_second_dense = core.Dense(5)
            deferred_sequential.add(deferred_second_dense)
            deferred_sequential(tf.constant([[1.0]]))
            status.run_restore_ops()
            self.assertAllEqual(
                [1.0, 2.0, 3.0, 4.0, 5.0],
                self.evaluate(deferred_second_dense.bias),
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
                optimizer = adam.Adam(0.001)
                root = tf.train.Checkpoint(
                    model=model
                )  # Do not save the optimizer with the checkpoint.
                optimizer_checkpoint = tf.train.Checkpoint(optimizer=optimizer)

                checkpoint_path = tf.train.latest_checkpoint(
                    checkpoint_directory
                )
                status = root.restore(save_path=checkpoint_path)
                input_value = tf.constant([[3.0]])

                def train_fn():
                    with tf.GradientTape() as tape:
                        loss = model(input_value)
                    variables = model.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    return optimizer.apply_gradients(zip(gradients, variables))

                if not tf.executing_eagerly():
                    train_fn = functools.partial(self.evaluate, train_fn())
                status.initialize_or_restore()
                # TODO(tanzheny): Add hyper variables to .variables(), and set
                # them with set_weights etc.
                variables_not_in_the_variables_property = [
                    obj
                    for obj in optimizer._hyper.values()
                    if isinstance(obj, tf.Variable)
                ]
                self.evaluate(
                    [
                        v.initializer
                        for v in optimizer.variables()
                        + variables_not_in_the_variables_property
                    ]
                )
                train_fn()
                model_save_path = root.save(file_prefix=checkpoint_prefix)
                self.evaluate(optimizer.beta_1.assign(42.0))
                optimizer_save_path = optimizer_checkpoint.save(
                    optimizer_only_prefix
                )
            del train_fn

            # Restore into a graph with the optimizer
            with test_utils.device(should_use_gpu=True):
                model = MyModel()
                optimizer = adam.Adam(0.001)
                root = tf.train.Checkpoint(optimizer=optimizer, model=model)
                status = root.restore(save_path=model_save_path)
                input_value = tf.constant([[3.0]])

                def train_fn1():
                    with tf.GradientTape() as tape:
                        loss = model(input_value)
                    variables = model.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    return optimizer.apply_gradients(zip(gradients, variables))

                if not tf.executing_eagerly():
                    train_fn1 = functools.partial(self.evaluate, train_fn1())
                status.initialize_or_restore()
                train_fn1()
                with self.assertRaises(AssertionError):
                    status.assert_existing_objects_matched()
                with self.assertRaises(AssertionError):
                    status.assert_consumed()
            del train_fn1

            # Make sure initialization doesn't clobber later restores
            with test_utils.device(should_use_gpu=True):
                model = MyModel()
                optimizer = adam.Adam(0.001, beta_1=1.0)
                root = tf.train.Checkpoint(optimizer=optimizer, model=model)
                opt_root = tf.train.Checkpoint(optimizer=optimizer)
                status = root.restore(save_path=model_save_path)
                init_only_optimizer_status = opt_root.restore(save_path=None)
                optimizer_status = opt_root.restore(
                    save_path=optimizer_save_path
                )
                input_value = tf.constant([[3.0]])

                def train_fn2():
                    with tf.GradientTape() as tape:
                        loss = model(input_value)
                    variables = model.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    return optimizer.apply_gradients(zip(gradients, variables))

                if not tf.executing_eagerly():
                    train_fn2 = functools.partial(self.evaluate, train_fn2())
                optimizer_status.run_restore_ops()
                status.initialize_or_restore()
                init_only_optimizer_status.initialize_or_restore()
                train_fn2()
                self.assertEqual(42.0, self.evaluate(optimizer.beta_1))


class _ManualScope(tf.Module):
    def __call__(self):
        with tf.compat.v1.variable_scope("ManualScope") as vs:
            self.variable_scope = vs
            with trackable_utils.capture_dependencies(template=self):
                return self._build()

    def _build(self):
        return tf.compat.v1.get_variable(name="in_manual_scope", shape=[])


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class TemplateTests(test_combinations.TestCase):
    def test_trackable_save_restore(self):
        with self.test_session():

            def _templated():
                v = tf.compat.v1.get_variable(
                    "v",
                    shape=[1],
                    initializer=tf.compat.v1.zeros_initializer(),
                    use_resource=True,
                )
                v2 = tf.compat.v1.get_variable(
                    "v2",
                    shape=[1],
                    initializer=tf.compat.v1.zeros_initializer(),
                    use_resource=True,
                )
                manual = _ManualScope()
                return v, v + 1.0, v2, manual, manual()

            save_template = tf.compat.v1.make_template("s1", _templated)
            v1_save, _, v2_save, manual_scope, manual_scope_v = save_template()
            self.assertEqual(
                set(
                    [
                        id(v1_save),
                        id(v2_save),
                        id(manual_scope),
                        id(manual_scope_v),
                        id(save_template),
                    ]
                ),
                set(map(id, trackable_utils.list_objects(save_template))),
            )
            self.assertDictEqual(
                {"in_manual_scope": manual_scope_v},
                manual_scope._trackable_children(),
            )
            optimizer = adam.Adam(0.0)
            save_root = tf.train.Checkpoint(
                my_template=save_template, optimizer=optimizer
            )
            optimizer.minimize(v1_save.read_value, var_list=[v1_save])
            self.evaluate([v.initializer for v in save_template.variables])
            optimizer_variables = optimizer.variables() + list(
                optimizer._hyper.values()
            )
            self.evaluate([v.initializer for v in optimizer_variables])
            self.evaluate(v1_save.assign([12.0]))
            self.evaluate(v2_save.assign([14.0]))
            checkpoint_directory = self.get_temp_dir()
            checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
            save_path = save_root.save(checkpoint_prefix)

            load_template = tf.compat.v1.make_template("s2", _templated)
            load_optimizer = adam.Adam(0.0)
            load_root = tf.train.Checkpoint(
                my_template=load_template, optimizer=load_optimizer
            )
            status = load_root.restore(save_path)
            var, var_plus_one, var2, _, _ = load_template()
            load_optimizer.minimize(var.read_value, var_list=[var])

            children = load_template._trackable_children()
            self.assertEqual({"v", "v2", "ManualScope"}, children.keys())
            status.assert_consumed().run_restore_ops()
            self.assertAllEqual([12.0], self.evaluate(var))
            self.assertAllEqual([13.0], self.evaluate(var_plus_one))
            self.assertAllEqual([14.0], self.evaluate(var2))


class CheckpointCompatibilityTests(test_combinations.TestCase):
    def _initialized_model(self):
        input_value = tf.constant([[3.0]])
        model = MyModel()
        optimizer = adam.Adam(0.001)
        root_trackable = tf.train.Checkpoint(optimizer=optimizer, model=model)
        with tf.GradientTape() as tape:
            loss = model(input_value)
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        self.evaluate(trackable_utils.gather_initializers(root_trackable))
        self.evaluate(train_op)
        # A regular variable, a slot variable, and a non-slot Optimizer variable
        # with known values to check when loading.
        self.evaluate(model._named_dense.bias.assign([1.0]))
        self.evaluate(
            optimizer.get_slot(
                var=model._named_dense.bias, slot_name="m"
            ).assign([2.0])
        )
        self.evaluate(optimizer.beta_1.assign(3.0))
        return root_trackable

    def _set_sentinels(self, root_trackable):
        self.evaluate(root_trackable.model._named_dense.bias.assign([101.0]))
        self.evaluate(
            root_trackable.optimizer.get_slot(
                var=root_trackable.model._named_dense.bias, slot_name="m"
            ).assign([102.0])
        )
        self.evaluate(root_trackable.optimizer.beta_1.assign(103.0))

    def _check_sentinels(self, root_trackable):
        self.assertAllEqual(
            [1.0], self.evaluate(root_trackable.model._named_dense.bias)
        )
        self.assertAllEqual(
            [2.0],
            self.evaluate(
                root_trackable.optimizer.get_slot(
                    var=root_trackable.model._named_dense.bias, slot_name="m"
                )
            ),
        )
        self.assertAllEqual(3.0, self.evaluate(root_trackable.optimizer.beta_1))

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
                    global_step=root.optimizer.iterations,
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
                status.assert_nontrivial_match()
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

    def testIgnoreSaveCounter(self):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        with self.cached_session() as session:
            # Create and save a model using Saver() before using a Checkpoint.
            # This generates a snapshot without the Checkpoint's `save_counter`.
            model = sequential.Sequential()
            model.add(reshaping.Flatten(input_shape=(1,)))
            model.add(core.Dense(1))
            name_saver = tf.compat.v1.train.Saver(model.trainable_variables)
            save_path = name_saver.save(
                sess=session, save_path=checkpoint_prefix, global_step=1
            )
            # Checkpoint.restore must successfully load that checkpoint.
            ckpt = tf.train.Checkpoint(model=model)
            status = ckpt.restore(save_path)
            status.assert_existing_objects_matched()
            # It should, however, refuse to load a checkpoint where an unrelated
            # `save_counter` variable is missing.
            model.layers[1].var = tf.Variable(0.0, name="save_counter")
            status = ckpt.restore(save_path)
            with self.assertRaises(AssertionError):
                status.assert_existing_objects_matched()


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.test.main()
