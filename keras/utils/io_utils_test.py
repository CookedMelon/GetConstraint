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
"""Tests for io_utils."""

import builtins
import sys
from pathlib import Path

import tensorflow.compat.v2 as tf

from keras.testing_infra import test_combinations
from keras.utils import io_utils


class TestIOUtils(test_combinations.TestCase):
    def test_ask_to_proceed_with_overwrite(self):
        with tf.compat.v1.test.mock.patch.object(builtins, "input") as mock_log:
            mock_log.return_value = "y"
            self.assertTrue(
                io_utils.ask_to_proceed_with_overwrite("/tmp/not_exists")
            )

            mock_log.return_value = "n"
            self.assertFalse(
                io_utils.ask_to_proceed_with_overwrite("/tmp/not_exists")
            )

            mock_log.side_effect = ["m", "y"]
            self.assertTrue(
                io_utils.ask_to_proceed_with_overwrite("/tmp/not_exists")
            )

            mock_log.side_effect = ["m", "n"]
            self.assertFalse(
                io_utils.ask_to_proceed_with_overwrite("/tmp/not_exists")
            )

    def test_path_to_string(self):
        class PathLikeDummy:
            def __fspath__(self):
                return "dummypath"

        dummy = object()
        # conversion of PathLike
        self.assertEqual(io_utils.path_to_string(Path("path")), "path")
        self.assertEqual(io_utils.path_to_string(PathLikeDummy()), "dummypath")

        # pass-through, works for all versions of python
        self.assertEqual(io_utils.path_to_string("path"), "path")
        self.assertIs(io_utils.path_to_string(dummy), dummy)

    def test_print_msg(self):
        enabled = io_utils.is_interactive_logging_enabled()

        io_utils.disable_interactive_logging()
        self.assertFalse(io_utils.is_interactive_logging_enabled())

        with self.assertLogs(level="INFO") as logged:
            io_utils.print_msg("Testing Message")
        self.assertIn("Testing Message", logged.output[0])

        io_utils.enable_interactive_logging()
        self.assertTrue(io_utils.is_interactive_logging_enabled())

        with self.captureWritesToStream(sys.stdout) as printed:
            io_utils.print_msg("Testing Message")
        self.assertEqual("Testing Message\n", printed.contents())

        if enabled:
            io_utils.enable_interactive_logging()
        else:
            io_utils.disable_interactive_logging()


if __name__ == "__main__":
    tf.test.main()
