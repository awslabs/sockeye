# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import tempfile
import os

import pytest

from sockeye import config


class ConfigTest(config.Config):
    yaml_tag = "!ConfigTest"

    def __init__(self, param, config=None):
        super().__init__()
        self.param = param
        self.config = config


def test_base_freeze():
    c = config.Config()
    c.param = 1
    assert c.param == 1
    c.freeze()
    with pytest.raises(AttributeError) as e:
        c.param = 2
    assert str(e.value) == "Cannot set 'param' in frozen config"


def test_freeze():
    c1 = ConfigTest(param=1)
    c2 = ConfigTest(param=3)
    c1.param = 2
    assert c1.param == 2
    c1.config = c2
    assert c2 == c1.config
    c1.config.param = 2
    assert c1.config.param == 2
    c1.freeze()
    assert c1.config._frozen  # pylint: disable= no-member
    assert c2._frozen  # pylint: disable= no-member
    with pytest.raises(AttributeError) as e:
        c1.param = 3
    assert str(e.value) == "Cannot set 'param' in frozen config"
    with pytest.raises(AttributeError) as e:
        c1.config.param = 3
    assert str(e.value) == "Cannot set 'param' in frozen config"


def test_config_repr():
    c1 = ConfigTest(param=1, config=ConfigTest(param=3))
    c1.config.freeze()
    assert str(c1) == "Config[_frozen=False, config=Config[_frozen=True, config=None, param=3], param=1]"


def test_eq():
    basic_c = config.Config()
    c1 = ConfigTest(param=1)
    c1_other = ConfigTest(param=1)
    c2 = ConfigTest(param=2)

    c_nested = ConfigTest(param=1, config=c1)
    c_nested_other = ConfigTest(param=1, config=c1_other)
    c_nested_c2 = ConfigTest(param=1, config=c2)

    assert c1 != "OTHER_TYPE"
    assert c1 != basic_c
    assert c1 == c1_other
    assert c1 != c2
    assert c_nested == c_nested_other
    assert c_nested != c_nested_c2


def test_no_self_attribute():
    c1 = ConfigTest(param=1)
    with pytest.raises(AttributeError) as e:
        c1.config = c1
    assert str(e.value) == "Cannot set self as attribute"


def test_serialization():
    c1 = ConfigTest(param=1, config=ConfigTest(param=2))
    expected_serialization = """!ConfigTest
config: !ConfigTest
  config: null
  param: 2
param: 1
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fname = os.path.join(tmp_dir, "config")
        c1.freeze()
        c1.save(fname)
        assert os.path.exists(fname)
        with open(fname) as f:
            assert f.read() == expected_serialization

        c2 = config.Config.load(fname)
        assert c2.param == c1.param
        assert c2.config.param == c1.config.param
        assert not c2._frozen


def test_copy():
    c1 = ConfigTest(param=1)
    copy_c1 = c1.copy()
    # should be a different object that is equal to the original object
    assert c1 is not copy_c1
    assert c1 == copy_c1

    # optionally you can modify attributes when copying:
    mod_c1 = ConfigTest(param=5)
    mod_copy_c1 = c1.copy(param=5)
    assert mod_c1 is not mod_copy_c1
    assert mod_c1 == mod_copy_c1
    assert c1 != mod_copy_c1


