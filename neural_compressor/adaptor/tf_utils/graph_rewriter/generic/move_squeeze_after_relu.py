#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from neural_compressor.utils.utility import dump_elapsed_time

from ..graph_base import GraphRewriterBase
from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer


class MoveSqueezeAfterReluOptimizer(GraphRewriterBase):
    def __init__(self, model):
        super().__init__(model)
        self.op_list = ['Relu', 'Sigmoid']

    @dump_elapsed_time("Pass MoveSqueezeAfterReluOptimizer")
    def do_transformation(self):
        g = GraphAnalyzer()
        g.graph = self.model
        graph_info = g.parse_graph()
        for node in self.model.node:
            if node.op in self.op_list and \
                    node.input[0] in graph_info and \
                        graph_info[node.input[0]].node.op == 'BiasAdd':
                biasadd_node = graph_info[node.input[0]].node
                biasadd_input = graph_info[biasadd_node.name].node.input[0]
                squeeze_node = graph_info[biasadd_input].node
                relu_output = graph_info[node.name].outputs
                if squeeze_node.op == 'Squeeze':
                    #biasadd
                    for i, input in enumerate(biasadd_node.input):
                        if input == biasadd_input:
                            new_input = biasadd_node.input[:i] + [squeeze_node.input[0]] + \
                                    biasadd_node.input[i+1:]
                            graph_info[biasadd_node.name].node.ClearField('input')
                            graph_info[biasadd_node.name].node.input.extend(new_input)
                            graph_info[squeeze_node.name].outputs.remove(biasadd_node.name)
                    #conv output
                    conv = squeeze_node.input[0]
                    conv_outputs = graph_info[conv].outputs
                    for i, output in enumerate(conv_outputs):
                        if output == squeeze_node.name:
                            graph_info[conv].outputs.remove(squeeze_node.name)
                            graph_info[conv].outputs.append(biasadd_node.name)
                    #squeeze input
                    squeeze_node.ClearField('input')
                    squeeze_node.input.extend([node.name])
                    #expand input,squeeze output
                    for output in relu_output:
                        for i, input in enumerate(graph_info[output].node.input):
                            if input == node.name:
                                new_input = graph_info[output].node.input[:i] + [squeeze_node.name] +\
                                graph_info[output].node.input[i+1:]
                                graph_info[squeeze_node.name].outputs.append(output)
                                graph_info[output].node.ClearField('input')
                                graph_info[output].node.input.extend(new_input)
        return g.dump_graph()
