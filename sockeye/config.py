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

import copy

import yaml


class Config(yaml.YAMLObject):
    """
    Base configuration object that supports freezing of members and YAML (de-)serialization.
    Actual Configuration should subclass this object.
    """
    yaml_tag = '!Config'

    def __init__(self):
        self.__add_frozen()

    def __setattr__(self, key, value):
        if hasattr(self, '_frozen') and getattr(self, '_frozen'):
            raise AttributeError("Cannot set '%s' in frozen config" % key)
        if value == self:
            raise AttributeError("Cannot set self as attribute")
        object.__setattr__(self, key, value)

    def freeze(self):
        """
        Freezes this Config object, disallowing modification or addition of any parameters.
        """
        if getattr(self, '_frozen'):
            return
        object.__setattr__(self, "_frozen", True)
        for k, v in self.__dict__.items():
            if isinstance(v, Config) and k != "self":
                v.freeze()  # pylint: disable= no-member

    def __repr__(self):
        return "Config[%s]" % ", ".join("%s=%s" % (str(k), str(v)) for k, v in sorted(self.__dict__.items()))

    def __del_frozen(self):
        """
        Removes _frozen attribute from this instance and all its child configurations.
        """
        self.__delattr__('_frozen')
        for attr, val in self.__dict__.items():
            if isinstance(val, Config) and hasattr(val, '_frozen'):
                val.__del_frozen()  # pylint: disable= no-member

    def __add_frozen(self):
        """
        Adds _frozen attribute to this instance and all its child configurations.
        """
        setattr(self, "_frozen", False)
        for attr, val in self.__dict__.items():
            if isinstance(val, Config):
                val.__add_frozen()  # pylint: disable= no-member

    def save(self, fname: str):
        """
        Saves this Config (without the frozen state) to a file called fname.

        :param fname: Name of file to store this Config in.
        """
        obj = copy.deepcopy(self)
        obj.__del_frozen()
        with open(fname, 'w') as out:
            yaml.dump(obj, out, default_flow_style=False)

    @staticmethod
    def load(fname: str) -> 'Config':
        """
        Returns a Config object loaded from a file. The loaded object is not frozen.

        :param fname: Name of file to load the Config from.
        :return: Configuration.
        """
        with open(fname) as inp:
            obj = yaml.load(inp)
            obj.__add_frozen()
            return obj
