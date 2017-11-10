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
import inspect

import yaml


class TaggedYamlObjectMetaclass(yaml.YAMLObjectMetaclass):
    def __init__(cls, name, bases, kwds):
        cls.yaml_tag = "!" + name
        new_kwds = {}
        new_kwds.update(kwds)
        new_kwds['yaml_tag'] = "!" + name
        super().__init__(name, bases, new_kwds)


class Config(yaml.YAMLObject, metaclass=TaggedYamlObjectMetaclass):
    """
    Base configuration object that supports freezing of members and YAML (de-)serialization.
    Actual Configuration should subclass this object.
    """
    def __init__(self):
        self.__add_frozen()

    def __setattr__(self, key, value):
        if hasattr(self, '_frozen') and getattr(self, '_frozen'):
            raise AttributeError("Cannot set '%s' in frozen config" % key)
        if value == self:
            raise AttributeError("Cannot set self as attribute")
        object.__setattr__(self, key, value)

    def __setstate__(self, state):
        """Pickle protocol implementation."""
        # We first take the serialized state:
        self.__dict__.update(state)
        # Then we take the constructors default values for missing arguments in order to stay backwards compatible
        # This way we can add parameters to Config objects and still load old models.
        init_signature = inspect.signature(self.__init__)
        for param_name, param in init_signature.parameters.items():
            if param.default is not param.empty:
                if not hasattr(self, param_name):
                    object.__setattr__(self, param_name, param.default)

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

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        for k, v in self.__dict__.items():
            if k != "self":
                if k not in other.__dict__:
                    return False
                if self.__dict__[k] != other.__dict__[k]:
                    return False
        return True

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

    def copy(self, **kwargs):
        """
        Create a copy of the config object, optionally modifying some of the attributes.
        For example `nn_config.copy(num_hidden=512)` will create a copy of `nn_config` where the attribute `num_hidden`
        will be set to the new value of num_hidden.

        :param kwargs:
        :return: A deep copy of the config object.
        """
        copy_obj = copy.deepcopy(self)
        for name, value in kwargs.items():
            object.__setattr__(copy_obj, name, value)
        return copy_obj
