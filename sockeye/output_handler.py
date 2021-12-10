# Copyright 2017--2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from itertools import chain
from typing import Optional

import sockeye.constants as C
from sockeye.utils import smart_open
from . import inference_pt


def get_output_handler(output_type: str,
                       output_fname: Optional[str] = None) -> 'OutputHandler':
    """

    :param output_type: Type of output handler.
    :param output_fname: Output filename. If none sys.stdout is used.
    :raises: ValueError for unknown output_type.
    :return: Output handler.
    """
    output_stream = sys.stdout if output_fname is None else smart_open(output_fname, mode='w')
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
    elif output_type == C.OUTPUT_HANDLER_TRANSLATION_WITH_FACTORS:
        return FactoredStringOutputHandler(output_stream)
    else:
        raise ValueError("unknown output type")


class OutputHandler(ABC):
    """
    Abstract output handler interface
    """

    @abstractmethod
    def handle(self,
               t_input: inference_pt.TranslatorInput,
               t_output: inference_pt.TranslatorOutput,
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
               t_input: inference_pt.TranslatorInput,
               t_output: inference_pt.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        print("%s" % t_output.translation, file=self.stream, flush=True)

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
               t_input: inference_pt.TranslatorInput,
               t_output: inference_pt.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        print("{:.6f}\t{}".format(t_output.score, t_output.translation), file=self.stream, flush=True)

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
               t_input: inference_pt.TranslatorInput,
               t_output: inference_pt.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        result = "{:.6f}".format(t_output.score)
        if hasattr(t_output, 'factor_scores') and t_output.factor_scores:
            factor_scores = "\t".join("{:.6f}".format(fs) for fs in t_output.factor_scores)
            result = f"{result}\t{factor_scores}"
        print(result, file=self.stream, flush=True)

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
               t_input: inference_pt.TranslatorInput,
               t_output: inference_pt.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        print("{:.6f}\t{}\t{}".format(t_output.score,
                                      C.TOKEN_SEPARATOR.join(t_input.tokens),
                                      t_output.translation), file=self.stream, flush=True)

    def reports_score(self) -> bool:
        return True


class BenchmarkOutputHandler(StringOutputHandler):
    """
    Output handler to write detailed benchmark information to a stream.
    """

    def handle(self,
               t_input: inference_pt.TranslatorInput,
               t_output: inference_pt.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        print("input=%s\toutput=%s\tinput_tokens=%d\toutput_tokens=%d\ttranslation_time=%0.4f" %
              (C.TOKEN_SEPARATOR.join(t_input.tokens),
               t_output.translation,
               len(t_input.tokens),
               len(t_output.tokens),
               t_walltime),
              file=self.stream, flush=True)

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
               t_input: inference_pt.TranslatorInput,
               t_output: inference_pt.TranslatorOutput,
               t_walltime: float = 0.):
        """
        Outputs a JSON object of the fields in the `TranslatorOutput` object.
        """
        d_ = t_output.json()
        print(json.dumps(d_, sort_keys=True), file=self.stream, flush=True)

    def reports_score(self) -> bool:
        return True


class FactoredStringOutputHandler(OutputHandler):
    """
    Returns a factored string if the model produces target factors. If there are no target factors the output
    is equivalent to StringOutputHandler
    """
    def __init__(self, stream):
        self.stream = stream

    def handle(self,
               t_input: inference_pt.TranslatorInput,
               t_output: inference_pt.TranslatorOutput,
               t_walltime: float = 0.):
        """
        :param t_input: Translator input.
        :param t_output: Translator output.
        :param t_walltime: Total walltime for translation.
        """
        factored_string = C.TOKEN_SEPARATOR.join(C.DEFAULT_FACTOR_DELIMITER.join(factors) for factors in
                                                 zip(*chain([t_output.tokens], t_output.factor_tokens)))
        print(factored_string, file=self.stream, flush=True)

    def reports_score(self) -> bool:
        return False
