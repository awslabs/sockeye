# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
import sys
from abc import ABC, abstractmethod
from typing import Optional

import sockeye.constants as C
from . import data_io
from . import inference


def get_output_handler(output_type: str,
                       output_fname: Optional[str] = None) -> 'OutputHandler':
    """

    :param output_type: Type of output handler.
    :param output_fname: Output filename. If none sys.stdout is used.
    :raises: ValueError for unknown output_type.
    :return: Output handler.
    """
    output_stream = sys.stdout if output_fname is None else data_io.smart_open(output_fname, mode='w')
    if output_type == C.OUTPUT_HANDLER_TRANSLATION:
        return StringOutputHandler(output_stream)
    elif output_type == C.OUTPUT_HANDLER_SCORE:
        return ScoreOutputHandler(output_stream)
    elif output_type == C.OUTPUT_HANDLER_PAIR_WITH_SCORE:
        return PairWithScoreOutputHandler(output_stream)
    elif output_type == C.OUTPUT_HANDLER_TRANSLATION_WITH_SCORE:
        return StringWithScoreOutputHandler(output_stream)
    elif output_type == C.OUTPUT_HANDLER_BENCHMARK:
        return BenchmarkOutputHandler(output_stream)
    elif output_type == C.OUTPUT_HANDLER_JSON:
        return JSONOutputHandler(output_stream)
    else:
        raise ValueError("unknown output type")


class OutputHandler(ABC):
    """
    Abstract output handler interface
    """

    @abstractmethod
    def handle(self,
               t_input: inference.TranslatorInput,
               t_output: inference.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total wall-clock time for translation.
        """
        pass

    @abstractmethod
    def reports_score(self) -> bool:
        """
        True if output_handler makes use of TranslatorOutput.score
        :return:
        """
        pass


class StringOutputHandler(OutputHandler):
    """
    Output handler to write translation to a stream

    :param stream: Stream to write translations to (e.g. sys.stdout).
    """

    def __init__(self, stream):
        self.stream = stream

    def handle(self,
               t_input: inference.TranslatorInput,
               t_output: inference.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        self.stream.write("%s\n" % t_output.translation)
        self.stream.flush()

    def reports_score(self) -> bool:
        return False


class StringWithScoreOutputHandler(OutputHandler):
    """
    Output handler to write translation score and translation to a stream. The score and translation
    string are tab-delimited.

    :param stream: Stream to write translations to (e.g. sys.stdout).
    """

    def __init__(self, stream):
        self.stream = stream

    def handle(self,
               t_input: inference.TranslatorInput,
               t_output: inference.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        self.stream.write("{:.6f}\t{}\n".format(t_output.score, t_output.translation))
        self.stream.flush()

    def reports_score(self) -> bool:
        return True


class ScoreOutputHandler(OutputHandler):
    """
    Output handler to write translation score to a stream.

    :param stream: Stream to write translations to (e.g., sys.stdout).
    """

    def __init__(self, stream):
        self.stream = stream

    def handle(self,
               t_input: inference.TranslatorInput,
               t_output: inference.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        self.stream.write("{:.6f}\n".format(t_output.score))
        self.stream.flush()

    def reports_score(self) -> bool:
        return True


class PairWithScoreOutputHandler(OutputHandler):
    """
    Output handler to write translation score along with sentence input and output (tab-delimited).

    :param stream: Stream to write translations to (e.g., sys.stdout).
    """

    def __init__(self, stream):
        self.stream = stream

    def handle(self,
               t_input: inference.TranslatorInput,
               t_output: inference.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        self.stream.write("{:.6f}\t{}\t{}\n".format(t_output.score,
                                                    C.TOKEN_SEPARATOR.join(t_input.tokens),
                                                    t_output.translation))
        self.stream.flush()

    def reports_score(self) -> bool:
        return True


class BenchmarkOutputHandler(StringOutputHandler):
    """
    Output handler to write detailed benchmark information to a stream.
    """

    def handle(self,
               t_input: inference.TranslatorInput,
               t_output: inference.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        self.stream.write("input=%s\toutput=%s\tinput_tokens=%d\toutput_tokens=%d\ttranslation_time=%0.4f\n" %
                          (" ".join(t_input.tokens),
                           t_output.translation,
                           len(t_input.tokens),
                           len(t_output.tokens),
                           t_walltime))
        self.stream.flush()

    def reports_score(self) -> bool:
        return False


class BeamStoringHandler(OutputHandler):
    """
    Output handler to store beam histories in JSON format.

    :param stream: Stream to write translations to (e.g. sys.stdout).
    """

    def __init__(self, stream):
        self.stream = stream

    def handle(self,
               t_input: inference.TranslatorInput,
               t_output: inference.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total wall-clock time for translation.
        """
        assert len(t_output.beam_histories) >= 1, "Translator output should contain beam histories."
        # If the sentence was max_len split, we may have more than one history
        for h in t_output.beam_histories:
            # Add the number of steps in each beam
            h["number_steps"] = len(h["predicted_tokens"])  # type: ignore
            # Some outputs can have more than one beam, add the id for bookkeeping
            h["id"] = t_output.sentence_id  # type: ignore
            self.stream.write("%s\n" % json.dumps(h, sort_keys=True))
        self.stream.flush()

    def reports_score(self) -> bool:
        return False


class JSONOutputHandler(OutputHandler):
    """
    Output single-line JSON objects.
    Carries over extra fields from the input.
    """
    def __init__(self, stream) -> None:
        self.stream = stream

    def handle(self,
               t_input: inference.TranslatorInput,
               t_output: inference.TranslatorOutput,
               t_walltime: float = 0.):
        """
        Outputs a JSON object of the fields in the `TranslatorOutput` object.
        """

        d_ = t_output.json()

        self.stream.write("%s\n" % json.dumps(d_, sort_keys=True))
        self.stream.flush()

    def reports_score(self) -> bool:
        return True
