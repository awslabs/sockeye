# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import inspect
from typing import List, Optional, Dict, Tuple, Any, Union

from parsimonious import Grammar, NodeVisitor

from . import convolution
from . import layers
from . import rnn
from . import utils

# TODO: parallel should take a list of parallel layers (instead of separating them by ->)! so basically a list of layer_chains...

# TODO: boolean argument parsing?

custom_seq_grammar = Grammar(r"""
network = layer_chain
layer_chain = layer more_layers
more_layers = sep_layer*
sep_layer = sep layer
sep = "->"
layer = meta_layer / parallel_layer / repeat_layer / subsample_layer / standard_layer 
open = "("
close = ")"
empty_paren = open close

repeat_layer = "repeat" open int comma layer_chain close
subsample_layer = "subsample" open optional_params layer_chain_sep layer_chain close

standard_layer = standard_layer_name optional_parenthesis_params
standard_layer_name = ~"[a-z_A-Z]+"

meta_layer = meta_layer_name open layer_chain close
meta_layer_name = "res" / "highway"

parallel_layer = parallel_name open layer_chain more_layer_chains close
parallel_name = "parallel_add" / "parallel"
layer_chain_sep = "|"
separated_layer_chain = layer_chain_sep layer_chain
more_layer_chains = separated_layer_chain*

optional_parenthesis_params = parenthesis_params?
parenthesis_params = open param maybe_more_params close
optional_params = params?
params = param maybe_more_params
maybe_more_params = comma_param*
comma_param = comma param
comma = ~", *"
optional_comma = comma?
param = kw_param / arg_param
kw_param = string "=" arg_param
arg_param = float / int / bool / string
string = ~"[a-z_0-9]+"
float = ~"-?[0-9]+\.([0-9]+)?"
int = ~"-?[0-9]+"
bool = "True" / "False"
""")


# TODO: better error messages?!
# TODO: create documentation for each layer...

class CustomSeqParser(NodeVisitor):
    def __init__(self):
        super().__init__()

    def visit_parenthesis_params(self, node, rest):
        open_paran, param, more_params, close_paran = rest
        return [param] + more_params

    def visit_params(self, node, rest):
        param, maybe_more_params = rest
        return [param] + maybe_more_params

    def visit_float(self, node, param):
        return float(node.text)

    def visit_int(self, node, param):
        return int(node.text)

    def visit_string(self, node, param):
        return node.text

    def visit_maybe_more_params(self, node, children):
        return children

    def visit_comma_param(self, node, comma_param):
        # split off the comma:
        comma, param = comma_param
        return param

    def visit_kw_param(self, node, kw_param):
        key, sep, value = kw_param
        return key, value

    def visit_arg_param(self, node, children):
        # param always has a single child, which is either numeric or string
        return children[0]

    def visit_param(self, node, children):
        # param always has a single child, which is either numeric or string
        return children[0]

    def visit_bool(self, node, children):
        if node.text == "True":
            return True
        elif node.text == "False":
            return False
        else:
            raise ValueError("%s is not a boolean." % node.text)

    def visit_optional_parenthesis_params(self, node, children):
        if len(children) == 0:
            # no parameters
            return None
        else:
            # parameters are given, return parameter list
            return children[0]

    def visit_optional_params(self, node, children):
        if len(children) == 0:
            # no parameters
            return None
        else:
            # parameters are given, return parameter list
            return children[0]

    def visit_standard_layer(self, node, name_and_params):
        name, params = name_and_params
        return {"name": name, "params": params}

    def visit_standard_layer_name(self, node, children):
        return node.text

    def visit_meta_layer(self, node, children):
        meta_layer_name, open_paran, layer_chain, close_paran = children
        return {"name": meta_layer_name, "layers": layer_chain}

    def visit_meta_layer_name(self, node, children):
        return node.text

    def visit_repeat_layer(self, node, children):
        name, open_paran, num, comma, layer_chain, close = children
        return {"name": "repeat", "num": num, "layers": layer_chain}

    def visit_subsample_layer(self, node, children):
        name, open_paran, optional_params, sep, layer_chain, close_paran = children
        return {"name": "subsample", "layers": layer_chain, "params": optional_params}

    def visit_layer(self, node, children):
        return children[0]

    def visit_layer_chain(self, node, layer_chain):
        layer, more_layers = layer_chain
        return [layer] + more_layers

    def visit_more_layers(self, node, children):
        return children

    def visit_sep_layer(self, node, sep_layer):
        sep, layer = sep_layer
        return layer

    def visit_more_layer_chains(self, node, children):
        return children

    def visit_parallel_layer(self, node, parallel_layer):
        name, open_paran, layer_chain, more_layer_chains, close_paran = parallel_layer
        return {"name": name[0].text, "layer_chains": [layer_chain] + more_layer_chains}

    def visit_separated_layer_chain(self, node, separated_layer_chain):
        layer_chain_sep, layer_chain = separated_layer_chain
        return layer_chain

    def generic_visit(self, node, visited_children):
        # print("generic_visit", len(visited_children), node, visited_children)
        return visited_children or node


ParsedLayer = Dict


# TODO: wrap parsimonious exceptions and try to make them more useful!
def parse(description: str) -> List[ParsedLayer]:
    """
    Parse an architecture definition.

    :param description: A nested layer-chain description such as 'rnn->ff(512)'.
    :return: The parsed layer configurations as dictionaries containing keys and values that depend on the individual
        layers. The only key common to all layers is the layer name 'name'.
    """
    parsed_layers = CustomSeqParser().visit(custom_seq_grammar.parse(description))
    return parsed_layers


class KwargsDefaultFiller:

    def __init__(self, default_dict):
        self.default_dict = default_dict

    def fill(self, name, func, params: Optional[List[Union[Any, Tuple[str,Any]]]]):
        param_names = list(inspect.signature(func).parameters)
        param_names = [name for name in param_names if name != 'self']
        if params is not None:
            args = [param for param in params if not isinstance(param, tuple)]
            kwargs = [param for param in params if isinstance(param, tuple)]
            utils.check_condition(len(args) <= len(param_names),
                                  "Too many parameters given. %d were given, but only %d are needed (%s) for layer %s." % (len(args), len(param_names), ", ".join(param_names), name))
            param_kwargs = {name: value for name, value in zip(param_names, args)}
            for name, value in kwargs:
                param_kwargs[name] = value
        else:
            param_kwargs = {}
        for default_name, default_value in self.default_dict.items():
            if default_name in param_names and default_name not in param_kwargs:
                param_kwargs[default_name] = default_value
        return param_kwargs


def _fill_and_create(kwargs_filler, name, func, args):
    return func(**kwargs_filler.fill(name, func.__init__, args))


# TODO: create documentation of the different layers we have available
def _create_layer_configs(default_kwargs_filler, parsed_layers: List[Dict]) -> Tuple[List[layers.LayerConfig], bool]:
    source_attention_present = False
    layer_configs = []
    for layer in parsed_layers:
        name = layer['name']
        # TODO: can we simplify this? Maybe have LayerConfigs register themselves
        if name == 'ff':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.FeedForwardLayerConfig, layer['params']))
        elif name == 'linear':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.LinearLayerConfig, layer['params']))
        elif name == 'id':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.IdentityLayerConfig, layer['params']))
        elif name == 'mh_dot_att':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.MultiHeadSourceAttentionLayerConfig, layer['params']))
            source_attention_present = True
        elif name == 'mh_dot_self_att':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.MultiHeadSelfAttentionLayerConfig, layer['params']))
        elif name == 'cnn':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, convolution.ConvolutionalLayerConfig, layer['params']))
        elif name == 'qrnn':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, convolution.QRNNLayerConfig, layer['params']))
        elif name == 'pool':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, convolution.PoolingLayerConfig, layer['params']))
        elif name == 'rnn':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, rnn.RecurrentLayerConfig, layer['params']))
        elif name == 'birnn':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, rnn.BidirectionalRecurrentLayerConfig, layer['params']))
        elif name == 'dropout':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.DropoutLayerConfig, layer['params']))
        elif name == 'act':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.ActivationLayerConfig, layer['params']))
        elif name == 'res':
            sub_layers = layer["layers"]
            sub_layer_configs, sub_layers_source_attention_present = _create_layer_configs(default_kwargs_filler,
                                                                                           sub_layers)
            source_attention_present = source_attention_present or sub_layers_source_attention_present
            layer_configs.append(layers.ResidualLayerConfig(layer_configs=sub_layer_configs))
        elif name == 'highway':
            sub_layers = layer["layers"]
            sub_layer_configs, sub_layers_source_attention_present = _create_layer_configs(default_kwargs_filler,
                                                                                           sub_layers)
            source_attention_present = source_attention_present or sub_layers_source_attention_present
            layer_configs.append(layers.HighwayLayerConfig(layer_configs=sub_layer_configs))
        elif name == 'repeat':
            num = layer["num"]
            sub_layers = layer["layers"]
            for i in range(0, num):
                sub_layer_configs, sub_layers_source_attention_present = _create_layer_configs(default_kwargs_filler,
                                                                                               sub_layers)
                source_attention_present = source_attention_present or sub_layers_source_attention_present
                layer_configs.extend(sub_layer_configs)
        elif name == 'pos':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.SinCosPositionalEmbeddingsLayerConfig, layer['params']))
        elif name == 'learn_pos':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.LearnedPositionalEmbeddingsLayerConfig, layer['params']))
        elif name == 'norm':
            layer_configs.append(_fill_and_create(default_kwargs_filler,
                                                  name, layers.LayerNormalizationLayerConfig, layer['params']))
        else:
            raise ValueError("Unknown layer %s." % name)
    return layer_configs, source_attention_present


# TODO: adapt doc-string
def parse_custom_seq_layers_description(default_num_hidden: int,
                                        default_num_embed: int,
                                        default_dropout: float,
                                        max_seq_len: int,
                                        description: str,
                                        source_attention_needed: bool,
                                        source_attention_forbidden: bool) -> List[layers.LayerConfig]:
    """

    :param num_hidden: The default number of hidden units, for all layers which do not specify them.
    :param description: A custom layer description string such as 'ff->cnn->self_att->ff'.
    :return: The parsed list of layer descriptions.
    """
    parsed_layers = CustomSeqParser().visit(custom_seq_grammar.parse(description))

    kwargs_filler = KwargsDefaultFiller({"dropout": default_dropout,
                                         "num_hidden": default_num_hidden,
                                         "num_embed": default_num_embed,
                                         "max_seq_len": max_seq_len})

    layer_configs, source_attention_present = _create_layer_configs(kwargs_filler, parsed_layers)

    if source_attention_needed:
        utils.check_condition(source_attention_present,
                              "At least one source attention mechanism needed.")
    if source_attention_forbidden:
        utils.check_condition(not source_attention_present,
                              "Source attention not allowed on the encoder side.")

    return layer_configs

