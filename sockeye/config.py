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
from typing import Any, Dict

import inspect
import jsonpickle

class Config:
    """
    Base configuration object that supports freezing of members and JSON (de-)serialization.
    Actual Configuration should subclass this object.
    """

    def __init__(self, arg_values) -> None:
        for i in inspect.getfullargspec(arg_values['self'].__init__).args[0:]:
            setattr(arg_values['self'], i, arg_values[i])
        setattr(arg_values['self'], "_frozen", False)
        # self._frozen = False

    def __setattr__(self, key, value):
        if hasattr(self, '_frozen') and self._frozen:
            raise AttributeError("Cannot set '%s' in frozen config" % key)
        object.__setattr__(self, key, value)

    def freeze(self):
        """
        Freezes this Config object, disallowing modification or addition of any parameters.
        """
        self._frozen = True
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                if k == "self":
                    continue
                v.freeze()

    def __repr__(self):
        return "Config%s" % str(self.__dict__)

    def save(self, fname: str):
        """
        Saves this Config to a file called fname.
        :param fname: Name of file to store this Config in.
        """
        jsonpickle.set_encoder_options('json', sort_keys=True, indent=1)
        with open(fname, 'w') as fout:
            fout.write(jsonpickle.encode(self))

    @staticmethod
    def load(fname: str) -> 'Config':
        """
        Returns a Config object loaded from a file.

        :param fname: Name of file to load the Config from.
        :return: Configuration.
        """
        with open(fname) as fin:
            return jsonpickle.decode(fin.read())
