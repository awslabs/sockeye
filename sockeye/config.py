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
    Base configuration object that supports freezing of members and JSON (de-)serialization.
    Actual Configuration should subclass this object.
    """
    yaml_tag = u'!Config'

    def __setattr__(self, key, value):
        if hasattr(self, '_frozen') and self._frozen:
            raise AttributeError("Cannot set '%s' in frozen config" % key)
        object.__setattr__(self, key, value)

    def freeze(self):
        """
        Freezes this Config object, disallowing modification or addition of any parameters.
        """
        if hasattr(self, '_frozen') and self._frozen:  # It's ok to freeze an already frozen config
            return
        self._frozen = True
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                v.freeze()

    def __repr__(self):
        return "Config[%s]" % ", ".join("%s=%s" %(str(k), str(v)) for k, v in sorted(self.__dict__.items()))

    def __del_frozen(self):
        self.__delattr__('_frozen')
        for attr, val in self.__dict__.items():
            if isinstance(val, Config) and hasattr(val, '_frozen'):
                val.__del_frozen()

    def __add_frozen(self):
        setattr(self, "_frozen", False)
        for attr, val in self.__dict__.items():
            if isinstance(val, Config):
                val.__add_frozen()

    def save(self, fname: str):
        """
        Saves this Config to a file called fname.

        :param fname: Name of file to store this Config in.
        """
        obj = copy.deepcopy(self)
        obj.__del_frozen()
        with open(fname, 'w') as fout:
            yaml.dump(obj, fout, default_flow_style=False)

    @staticmethod
    def load(fname: str) -> 'Config':
        """
        Returns a Config object loaded from a file.

        :param fname: Name of file to load the Config from.
        :return: Configuration.
        """
        with open(fname) as fin:
            obj = yaml.load(fin)
            obj.__add_frozen()
            return obj
