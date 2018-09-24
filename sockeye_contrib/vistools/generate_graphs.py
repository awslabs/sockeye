# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Modified by Alexander Rush, 2017

# MIT License
#
# Copyright (c) 2017-present The OpenNMT Authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Generate beam search visualization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import json
import shutil
from string import Template

import networkx as nx
from networkx.readwrite import json_graph


# Pad token used in sockeye
# Used to filter out pad tokens from the graph
PAD_TOKEN = "<pad>"

HTML_TEMPLATE = Template("""
                         <!DOCTYPE html>
                         <html lang="en">
                         <head>
                         <meta charset="utf-8">
                         <title>$SENT - Beam Search</title>
                         <link rel="stylesheet" type="text/css" href="tree.css">
                         <script src="http://d3js.org/d3.v3.min.js"></script>
                         </head>
                         <body>
                         <script>
                         var treeData = $DATA
                         </script>
                         <script src="tree.js"></script>
                         </body>
                         </html>""")


def _add_graph_level(graph, level, parent_ids, names, scores, normalized_scores,
                     include_pad):
    """Adds a level to the passed graph"""
    for i, parent_id in enumerate(parent_ids):
        if not include_pad and names[i] == PAD_TOKEN:
            continue
        new_node = (level, i)
        parent_node = (level - 1, parent_id)
        raw_score = '%.3f' % float(scores[i]) if scores[i] is not None else '-inf'
        norm_score = '%.3f' % float(normalized_scores[i]) if normalized_scores[i] is not None else '-inf'

        graph.add_node(new_node)
        graph.node[new_node]["name"] = names[i]
        graph.node[new_node]["score"] = "[RAW] {}".format(raw_score)
        graph.node[new_node]["norm_score"] = "[NORM] {}".format(norm_score)
        graph.node[new_node]["size"] = 100
        # Add an edge to the parent
        graph.add_edge(parent_node, new_node)

def create_graph(predicted_ids, parent_ids, scores, normalized_scores, include_pad):

    seq_length = len(predicted_ids)
    graph = nx.DiGraph()
    for level in range(seq_length):
        names = [pred for pred in predicted_ids[level]]
        _add_graph_level(graph, level + 1, parent_ids[level], names,
                         scores[level], normalized_scores[level], include_pad)
    graph.node[(0, 0)]["name"] = "START"
    return graph

def generate(input_data, output_dir, include_pad=False):

    path_base = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Copy required files
    shutil.copy2(path_base+"/templates/tree.css", output_dir)
    shutil.copy2(path_base+"/templates/tree.js", output_dir)

    with open(input_data) as beams:
        for i, line in enumerate(beams):
            beam = json.loads(line)

            graph = create_graph(predicted_ids=beam["predicted_tokens"],
                                 parent_ids=beam["parent_ids"],
                                 scores=beam["scores"],
                                 normalized_scores=beam["normalized_scores"],
                                 include_pad=include_pad)

            json_str = json.dumps(
                json_graph.tree_data(graph, (0, 0)),
                ensure_ascii=True)

            html_str = HTML_TEMPLATE.substitute(DATA=json_str, SENT=str(i))
            output_path = os.path.join(output_dir, "{:06d}.html".format(i))
            with open(output_path, "w", encoding="utf-8") as out:
                out.write(html_str)
    print("Output beams written to: {}".format(output_dir))

def main():
    parser = argparse.ArgumentParser(description="Generate beam search visualizations")
    parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="path to the beam search data file")
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="path to the output directory")
    parser.add_argument('--pad', dest='include_pad', action='store_true')
    parser.add_argument('--no-pad', dest='include_pad', action='store_false')
    parser.set_defaults(include_pad=False)
    args = parser.parse_args()

    generate(args.data, args.output_dir, include_pad=args.include_pad)


if __name__ == "__main__":
    main()
