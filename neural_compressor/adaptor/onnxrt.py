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
# pylint: disable=no-member

import os
import copy
import logging
from collections import OrderedDict
from collections.abc import KeysView
import yaml
import numpy as np
from packaging.version import Version
from neural_compressor.adaptor.adaptor import adaptor_registry, Adaptor
from neural_compressor.adaptor.query import QueryBackendCapability
from neural_compressor.utils.utility import LazyImport, dump_elapsed_time, \
                                            GLOBAL_STATE, MODE
from neural_compressor.utils.utility import Statistics
from neural_compressor.experimental.data.dataloaders.base_dataloader import BaseDataLoader
from neural_compressor.conf.dotdict import deep_get
import math

onnx = LazyImport("onnx")
ort = LazyImport("onnxruntime")
ort_ext = LazyImport("onnxruntime_extensions")
ONNXRT152_VERSION = Version("1.5.2")
ONNXRT170_VERSION = Version("1.7.0")
ONNXRT112_VERSION = Version("1.12.0")

logger = logging.getLogger()

class ONNXRTAdaptor(Adaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)
        self.__config_dict = {}
        self.quantizable_ops = []
        self.device = framework_specific_info["device"]
        self.static = framework_specific_info["approach"] == "post_training_static_quant"
        self.backend = framework_specific_info["backend"]
        self.work_space = framework_specific_info["workspace_path"]
        self.graph_optimization = framework_specific_info["graph_optimization"]
        self.recipes = deep_get(framework_specific_info, 'recipes', {})
        self.benchmark = (GLOBAL_STATE.STATE == MODE.BENCHMARK)
        os.makedirs(self.work_space, exist_ok=True)
        self.pre_optimized_model = None
        self.quantizable_op_types = []
        for precision in self.query_handler.get_precisions():
            if precision != 'fp32':
                if self.device == 'cpu' and precision == 'fp16':
                    continue
                self.quantizable_op_types += \
                    self.query_handler.get_op_types_by_precision(precision=precision)
 
        self.evaluate_nums = 0

        self.fp32_results = []
        self.fp32_preds_as_label = False
        self.quantize_config = {} # adaptor should know current configs at any time
        self.quantize_params = {} # adaptor should know current params at any time
        self.min_max = None

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """The function is used to do calibration and quanitization in post-training
           quantization.

        Args:
            tune_cfg (dict):     quantization config.
            model (object):      model need to do quantization.
            data_loader (object): calibration dataset.
            q_func (optional):   training function for quantization aware training mode,
                                 unimplement yet for onnx.

        Returns:
            (dict): quantized model
        """
        assert q_func is None, "quantization aware training has not been supported on ONNXRUNTIME"
        model = self.pre_optimized_model if self.pre_optimized_model else model
        ort_version = Version(ort.__version__)
        if ort_version < ONNXRT152_VERSION: # pragma: no cover
            logger.warning("Quantize input needs onnxruntime 1.5.2 or newer.")
            return model
        if model.model.opset_import[0].version < 11: # pragma: no cover
            logger.warning("Quantize input needs model opset 11 or newer.")
        from neural_compressor.adaptor.ox_utils.util import QuantizationMode
        if self.backend in ["qlinearops", "qoperator"]:
            backend = QuantizationMode.QLinearOps
            if self.backend == "qlinearops":
                logger.warning("onnxrt_qlinearops uses the same model representation format as "
                               "onnxrt_qoperator. Recommended to use onnxrt_qoperator to align "
                               "with ONNXRUNTIME QuantFormat")
        elif self.backend == "qdq":
            assert ort_version >= ONNXRT170_VERSION, 'QDQ mode needs onnxruntime1.7.0 or newer'
            backend = "qdq"
        else:
            backend = QuantizationMode.IntegerOps

        self.quantizable_ops = self._query_quantizable_ops(model.model)
        tmp_model = copy.deepcopy(model)

        quantize_config = self._cfg_to_quantize_config(tune_cfg)
        iterations = tune_cfg.get('calib_iteration', 1)
        calib_sampling_size = tune_cfg.get('calib_sampling_size', 1)
        if self.static:
            if isinstance(data_loader, BaseDataLoader):
                batch_size = data_loader.batch_size
                try:
                    for i in range(batch_size):
                        if calib_sampling_size % (batch_size - i) == 0:
                            calib_batch_size = batch_size - i
                            if i != 0:  # pragma: no cover
                                logger.warning("Reset `calibration.dataloader.batch_size` field "
                                               "to {}".format(calib_batch_size) +
                                               " to make sure the sampling_size is "
                                               "divisible exactly by batch size")
                            break
                    tmp_iterations = int(math.ceil(calib_sampling_size / calib_batch_size))
                    data_loader.batch(calib_batch_size)
                    quantize_params = self._get_quantize_params(tmp_model.model, data_loader, \
                                                                quantize_config, tmp_iterations)
                except Exception as e:  # pragma: no cover
                    if 'Got invalid dimensions for input' in str(e):
                        logger.warning("Please set sampling_size to a multiple of {}".format(
                            str(e).partition('Expected: ')[2].partition('\n')[0]))
                        exit(0)
                    logger.warning(
                        "Fail to forward with batch size={}, set to {} now.".
                        format(batch_size, 1))
                    data_loader.batch(1)
                    quantize_params = self._get_quantize_params(tmp_model.model, data_loader, \
                                                                quantize_config, calib_sampling_size)
            else:  # pragma: no cover
                if hasattr(data_loader, 'batch_size') and \
                  calib_sampling_size % data_loader.batch_size != 0:
                    logger.warning(
                        "Please note that calibration sampling size {} " \
                        "isn't divisible exactly by batch size {}. " \
                        "So the real sampling size is {}.".
                        format(calib_sampling_size, data_loader.batch_size,
                               data_loader.batch_size * iterations))
                quantize_params = self._get_quantize_params(tmp_model.model, data_loader, \
                                                            quantize_config, iterations)
        else:
            quantize_params = None
        self.quantize_params = quantize_params
        from neural_compressor.adaptor.ox_utils.quantizer import Quantizer
        quantizer = Quantizer(tmp_model.model,
            quantize_config,
            backend,
            self.static,
            quantize_params,
            self.quantizable_op_types,
            self.query_handler.get_fallback_list())
        quantizer.quantize_model()
        tmp_model.q_config = self._generate_qconfig(model.model, tune_cfg, quantize_params)
        tmp_model.model = quantizer.model.model
        self.quantize_config = quantize_config # update so other methods can know current configs

        self._dump_model_op_stats(tmp_model)
        tmp_model.topological_sort()
        return tmp_model

    def _generate_qconfig(self, model, tune_cfg, quantize_params):
        tune_cfg = copy.deepcopy(tune_cfg)
        for node in model.graph.node:
            if (node.name, node.op_type) not in tune_cfg['op']:
                continue
            scale_info = {}
            if quantize_params:
                for input_name in node.input:
                    if input_name in quantize_params:
                        scale_info[input_name] = quantize_params[input_name]
                for output_name in node.output:
                    if output_name in quantize_params:
                        scale_info[output_name] = quantize_params[output_name]
            tune_cfg['op'][(node.name, node.op_type)]['scale_info'] = scale_info
        fwk_info = {}
        fwk_info['approach'] = "post_training_static_quant" if self.static else \
                                                        "post_training_dynamic_quant"
        fwk_info['backend'] = self.backend
        fwk_info['workspace_path'] = self.work_space
        fwk_info['graph_optimization'] = self.graph_optimization
        fwk_info['device'] = self.device
        tune_cfg['framework_specific_info'] = fwk_info
        return tune_cfg

    @dump_elapsed_time("Pass recover model")
    def recover(self, model, q_config):
        """Execute the recover process on the specified model.

        Args:
            model (object):  model need to do quantization.
            q_config (dict): recover configuration

        Returns:
            (dict): quantized model
        """
        self._pre_optimize(model)
        model = self.pre_optimized_model
        ort_version = Version(ort.__version__)
        if ort_version < ONNXRT152_VERSION: # pragma: no cover
            logger.warning("Quantize input needs onnxruntime 1.5.2 or newer.")
            return model
        if model.model.opset_import[0].version < 11: # pragma: no cover
            logger.warning("Quantize input needs model opset 11 or newer.")

        from neural_compressor.adaptor.ox_utils.util import QuantizationMode
        if self.backend in ["qlinearops", "qoperator"]:
            backend = QuantizationMode.QLinearOps
        elif self.backend == "qdq":
            assert ort_version >= ONNXRT170_VERSION, 'QDQ mode needs onnxruntime1.7.0 or newer'
            backend = self.backend
        else:
            backend = QuantizationMode.IntegerOps
        from neural_compressor.adaptor.ox_utils.quantizer import Quantizer
        self.quantizable_ops = self._query_quantizable_ops(model.model)
        quantize_params, tune_cfg = self._parse_qconfig(q_config)
        quantize_config = self._cfg_to_quantize_config(tune_cfg)
        quantizer = Quantizer(model.model,
            quantize_config,
            backend,
            self.static,
            quantize_params,
            self.quantizable_op_types,
            self.query_handler.get_fallback_list())
 
        quantizer.quantize_model()
        model.model = quantizer.model.model
        model.topological_sort()
        return model

    def _parse_qconfig(self, q_config):
        quantize_params = {}
        tune_cfg = {}
        for k, v in q_config.items():
            if k == 'op':
                tune_cfg['op'] = {}
                for op_name_type, op_info in v.items():
                    node_dict = {}
                    for info_name, info_content in op_info.items():
                        if info_name != 'scale_info':
                            node_dict[info_name] = info_content
                        else:
                            for tensor_name, param in info_content.items():
                                quantize_params[tensor_name] = param
                    tune_cfg['op'][op_name_type] = node_dict
            else:
                tune_cfg[k] = v
        if len(quantize_params) == 0:
            quantize_params = None
        return quantize_params, tune_cfg

    def _dump_model_op_stats(self, model):
        fp32_op_list = []
        for precision in self.query_handler.get_precisions():
            if precision != 'fp32':
                fp32_op_list += self.query_handler.get_op_types_by_precision(precision=precision)
        qdq_ops = ["QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"]
        res = {}
        for op_type in fp32_op_list:
            res[op_type] = {'INT8':0, 'BF16': 0, 'FP16': 0, 'FP32':0}
        for op_type in qdq_ops:
            res[op_type] = {'INT8':0, 'BF16': 0, 'FP16': 0, 'FP32':0}

        for node in model.model.graph.node:
            if node.name.endswith('_quant'):
                if self.backend in ["qlinearops", "qdq", "qoperator"]:
                    origin_op_type = node.op_type.split('QLinear')[-1]
                else:
                    origin_op_type = node.op_type.split('Integer')[0]

                if origin_op_type in ["QAttention", "QGemm"]:
                    origin_op_type = origin_op_type[1:]
                elif origin_op_type == "DynamicQuantizeLSTM":
                    origin_op_type = "LSTM"
                elif origin_op_type == "QEmbedLayerNormalization":
                    origin_op_type = "EmbedLayerNormalization"
                res[origin_op_type]['INT8'] += 1

            elif node.op_type in qdq_ops:
                res[node.op_type]['INT8'] += 1

            elif node.op_type in fp32_op_list and node.name in self.quantize_config:
                if self.quantize_config[node.name] not in self.query_handler.get_fallback_list():
                    res[node.op_type]['FP32'] += 1
                else:
                    res[node.op_type][self.quantize_config[node.name].upper()] += 1

            elif node.op_type in res:
                res[node.op_type]['FP32'] += 1

        output_data = [[op_type, sum(res[op_type].values()), res[op_type]['INT8'],
            res[op_type]['BF16'], res[op_type]['FP16'], res[op_type]['FP32']] for \
            op_type in res.keys()]
        Statistics(output_data, 
                   header='Mixed Precision Statistics',
                   field_names=["Op Type", "Total", "INT8", "BF16", "FP16", "FP32"]).print_stat()

    def _get_quantize_params(self, model, data_loader, quantize_config, iterations):
        from neural_compressor.adaptor.ox_utils.onnxrt_mid import ONNXRTAugment
        from neural_compressor.model.onnx_model import ONNXModel
        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)
        black_nodes = [node for node in quantize_config if quantize_config[node]=='fp32']
        white_nodes = [node for node in quantize_config if quantize_config[node]!='fp32']
        augment = ONNXRTAugment(model, \
                  data_loader, self.quantizable_op_types, \
                  os.path.join(self.work_space, 'augmented_model.onnx'), \
                  black_nodes=black_nodes, white_nodes=white_nodes, \
                  iterations=list(range(0, quantize_config['calib_iteration'])))
        self.min_max = augment.dump_minmax()
        quantize_params = augment.dump_calibration()
        return quantize_params

    def inspect_tensor(self, model, dataloader, op_list=[],
                       iteration_list=[],
                       inspect_type='activation',
                       save_to_disk=False,
                       save_path=None,
                       quantization_cfg=None):
        '''The function is used by tune strategy class for dumping tensor info.
        '''
        from neural_compressor.adaptor.ox_utils.onnxrt_mid import ONNXRTAugment
        from neural_compressor.model.onnx_model import ONNXModel
        from neural_compressor.utils.utility import dump_data_to_local
        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)

        if len(op_list) > 0 and isinstance(op_list, KeysView):
            op_list = [item[0] for item in op_list]
        augment = ONNXRTAugment(model, dataloader, [], \
                  os.path.join(self.work_space, 'augment_for_inspect.onnx'), \
                  iterations=iteration_list,
                  white_nodes=op_list)
        tensors = augment.dump_tensor(activation=(inspect_type!='weight'),
                                      weight=(inspect_type!='activation'))
        if save_to_disk:
            if not save_path:
                save_path = self.work_space
            dump_data_to_local(tensors, save_path, 'inspect_result.pkl')
        return tensors

    def set_tensor(self, model, tensor_dict):
        from onnx import numpy_helper
        from neural_compressor.model.onnx_model import ONNXModel
        from neural_compressor.adaptor.ox_utils.util import quantize_data_with_scale_zero
        from neural_compressor.adaptor.ox_utils.util import quantize_data_per_channel
        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)
        assert "QuantizeLinear" in [node.op_type for node in model.model.graph.node], \
                                           'adaptor.set_tensor only accept int8 model'
        input_name_to_nodes = model.input_name_to_nodes
        for tensor_name, tensor_value in tensor_dict.items():
            if not tensor_name.endswith('_quantized'):
                tensor_name += '_quantized'
            not_filter = False
            scale_tensor, zo_tensor = model.get_scale_zero(tensor_name)
            if scale_tensor is None or zo_tensor is None:
                not_filter = True
            else:
                scale_value = numpy_helper.to_array(scale_tensor)
                zo_value = numpy_helper.to_array(zo_tensor)
            assert len(input_name_to_nodes[tensor_name]) == 1, \
                    'quantized filter weight should be input of only one node'
            node = input_name_to_nodes[tensor_name][0] #TBD only for conv bias
            node_name = node.name.replace('_quant', '')
            assert node_name in self.quantize_config
            q_type = self.quantize_config[node_name]['weight']['dtype']
            if not_filter:
                new_tensor_value = self._requantize_bias(model, tensor_name, tensor_value)
            elif self.quantize_config[node_name]['weight']['granularity'] == 'per_tensor':
                new_tensor_value = quantize_data_with_scale_zero(
                    tensor_value,
                    q_type,
                    self.quantize_config[node_name]['weight']['scheme'],
                    scale_value,
                    zo_value)
            elif (Version(ort.__version__) >= ONNXRT112_VERSION and \
                model.model.opset_import[0].version < 13) and \
                len(scale_tensor.dims) in [1, 2]:
                logger.warning("Skip setting per-channel quantized tensor {}, please " \
                    "use onnxruntime < 1.12.0 or upgrade model opset version to 13 or " \
                    "higher".format(tensor_name))
                return model
            else:
                new_tensor_value = quantize_data_per_channel(
                    tensor_value,
                    q_type,
                    self.quantize_config[node_name]['weight']['scheme'],
                    scale_value,
                    zo_value)
            model.set_initializer(tensor_name, new_tensor_value)
        return model

    def _requantize_bias(self, model, bias_name, bias_data):
        ''' helper function to requantize bias, borrowed from onnx_quantizer '''
        from onnx import numpy_helper
        node = model.input_name_to_nodes[bias_name][0]
        input_scale_name = node.input[1]
        input_scale = numpy_helper.to_array(model.get_initializer(input_scale_name))

        weight_scale_name = node.input[4]
        weight_scale = numpy_helper.to_array(model.get_initializer(weight_scale_name))

        bias_scale = input_scale * weight_scale
        new_bias_data = (bias_data / bias_scale).round().astype(np.int32)
        return new_bias_data

    def _pre_optimize(self, model, level=1):
        from neural_compressor.adaptor.ox_utils.util import \
            remove_init_from_model_input, split_shared_bias
        remove_init_from_model_input(model)
        sess_options = ort.SessionOptions()
        level = self.query_handler.get_graph_optimization()
        if self.graph_optimization.level:
            optimization_levels = {
                    'DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                    'ENABLE_BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    'ENABLE_EXTENDED': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    'ENABLE_ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL}
            assert self.graph_optimization.level in optimization_levels, "the optimization \
                                      choices are {}".format(optimization_levels.keys())
 
            level = optimization_levels[self.graph_optimization.level]
        sess_options.graph_optimization_level = level
        sess_options.optimized_model_filepath = os.path.join(self.work_space, \
            "Optimized_model.onnx")
        sess_options.register_custom_ops_library(ort_ext.get_library_path())
        _ = ort.InferenceSession(model.model.SerializeToString(), sess_options)
        tmp_model = onnx.load(sess_options.optimized_model_filepath)
        model.model = self._replace_gemm_with_matmul(tmp_model).model \
            if self.graph_optimization.gemm2matmul else tmp_model
        model.model = self._rename_node(model.model)
        model = self._revert_fusedconv(model)
        model = split_shared_bias(model)
        model.topological_sort()
        self.pre_optimized_model = model

    def _revert_fusedconv(self, model):
        from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg
        from onnx import onnx_pb as onnx_proto
        new_nodes = []
        remove_nodes = []
        for node in model.model.graph.node:
            if node.op_type == 'FusedConv':
                kwargs = {}
                activation_params = None
                for attr in node.attribute:
                    if attr.name == 'activation':
                        activation_type = attr.s.decode('utf-8')
                    elif attr.name == 'activation_params':
                        continue
                    else:
                        kwargs.update(attribute_to_kwarg(attr))
                if activation_type in ['Relu', 'Clip']:
                    continue
                conv = onnx.helper.make_node(
                    'Conv', node.input, [node.name], node.name.split('fused ')[-1], **kwargs)
                activation_input = conv.output

                activation = onnx.helper.make_node(activation_type,
                    conv.output, node.output, '_'.join((conv.name, activation_type)))
                new_nodes.extend([conv, activation])
                remove_nodes.append(node)
        model.model.graph.node.extend(new_nodes)
        for node in remove_nodes:
            model.model.graph.node.remove(node)
        model.update()
        return model

    def _rename_node(self, model):
        node_names = [i.name for i in model.graph.node]
        if len(set(node_names)) < len(node_names):
            logger.warning("This model has nodes with the same name, please check \
                renamed_model.onnx in workspace_path (default is nc_workspace) \
                for newly generated node name")
            for idx, node in enumerate(model.graph.node):
                if node_names.count(node.name) > 1:
                    node.name = node.op_type + '_nc_rename_' + str(idx)
            onnx.save(model, os.path.join(self.work_space, "renamed_model.onnx")) 
        return model

    @staticmethod
    def _replace_gemm_with_matmul(model):
        new_nodes = []
        from onnx import numpy_helper
        from neural_compressor.model.onnx_model import ONNXModel
        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)

        for node in model.nodes():
            if node.op_type == 'Gemm':
                alpha = 1.0
                beta = 1.0
                transA = 0
                transB = 0
                for attr in node.attribute:
                    if attr.name == 'alpha':
                        alpha = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'beta':
                        beta = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transA':
                        transA = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transB':
                        transB = onnx.helper.get_attribute_value(attr)
                if alpha == 1.0 and beta == 1.0 and transA == 0:
                    inputB = node.input[1]
                    if transB == 1:
                        B = model.get_initializer(node.input[1])
                        if B:
                            # assume B is not used by any other node
                            B_array = numpy_helper.to_array(B)
                            B_trans = numpy_helper.from_array(B_array.T)
                            B_trans.name = B.name
                            model.remove_initializer(B)
                            model.add_initializer(B_trans)

                            #TBD this is for onnx model zoo, which are all in old IR version
                            if model.model.ir_version < 4:
                                for input in model.model.graph.input:
                                    if input.name == B_trans.name:
                                        for i, dim in enumerate(input.type.tensor_type.shape.dim):
                                            dim.dim_value = B_array.T.shape[i]

                        else:
                            inputB += '_Transposed'
                            transpose_node = onnx.helper.make_node('Transpose',
                                                                inputs=[node.input[1]],
                                                                outputs=[inputB],
                                                                name=node.name+'_Transpose')
                            new_nodes.append(transpose_node)

                    matmul_node = onnx.helper.make_node('MatMul',
                            inputs=[node.input[0], inputB],
                            outputs=[node.output[0] + ('_MatMul' if len(node.input)>2 else '')],
                            name=node.name + '_MatMul')
                    new_nodes.append(matmul_node)

                    if len(node.input) > 2:
                        add_node = onnx.helper.make_node('Add',
                            inputs=[node.output[0] + '_MatMul', node.input[2]],
                            outputs=node.output,
                            name=node.name + '_Add')
                        new_nodes.append(add_node)

                # unsupported
                else:
                    new_nodes.append(node)

            # not GEMM
            else:
                new_nodes.append(node)

        model.graph().ClearField('node')
        model.graph().node.extend(new_nodes)

        return model

    def query_fw_capability(self, model):
        """The function is used to query framework capability.
        TODO: will be replaced by framework query API

        Args:
            model: onnx model

        Returns:
            (dict): quantization capability
        """
        # optype_wise and op_wise capability
        self._pre_optimize(model)
        exclude_first_quantizable_op = True if 'first_conv_or_matmul_quantization' in \
            self.recipes and not self.recipes['first_conv_or_matmul_quantization'] \
            else False
 
        precisions = self.query_handler.get_precisions()
        optype_wise = OrderedDict()
        quantizable_optype = set([i.op_type for i in self.pre_optimized_model.nodes()])

        op_wise = OrderedDict()
        for precision in precisions:
            if precision == 'fp16' and self.device == 'cpu' and os.getenv('FORCE_FP16') != '1':
                continue
            if precision in self.query_handler.get_quantization_capability():
                special_config_types = list(self.query_handler.get_quantization_capability() \
                    [precision].keys())
                default_config = self.query_handler.get_quantization_capability() \
                    [precision]['default']
            else:
                special_config_types = {}
                default_config = {'weight': {'dtype': [precision]}, 
                                  'activation': {'dtype': [precision]}}
            optypes = self.query_handler.get_op_types_by_precision(precision) if \
                self.query_handler.get_op_types_by_precision(precision) != ['*'] else \
                optype_wise.keys()
            for op in optypes:
                if op not in quantizable_optype:
                    continue
                if op not in special_config_types:
                    op_capability = default_config
                else:
                    op_capability = self.query_handler.get_quantization_capability() \
                        [precision][op]
                if op not in optype_wise.keys():
                    optype_wise[op] = copy.deepcopy(op_capability)
                elif precision not in optype_wise[op]['weight']['dtype']:
                    optype_wise[op]['weight']['dtype'].append(precision)
                    optype_wise[op]['activation']['dtype'].append(precision)

        for key, val in optype_wise.items():
            optype_wise[key]['weight']['dtype'] = \
                [i for i in precisions if i in val['weight']['dtype']]
            optype_wise[key]['activation']['dtype'] = \
                [i for i in precisions if i in val['activation']['dtype']]

        for _, node in enumerate(self.pre_optimized_model.nodes()):
            if node.op_type in optype_wise:
                if exclude_first_quantizable_op and node.op_type in ['Conv', 'MatMul']:
                    exclude_first_quantizable_op = False
                    tmp_cfg = copy.deepcopy(optype_wise[node.op_type])
                    for k, v in tmp_cfg.items():
                        v['dtype'] = list(filter(lambda x: x not in ['uint8', 'int8'], v['dtype']))
                    op_wise.update({(node.name, node.op_type): tmp_cfg})
                    continue
                op_wise.update(
                    {(node.name, node.op_type): copy.deepcopy(optype_wise[node.op_type])})
        return {'optypewise': optype_wise, 'opwise': op_wise}

    def _cfg_to_quantize_config(self, tune_cfg):
        quantize_config = {}
        quantize_config['calib_iteration'] = tune_cfg['calib_iteration']
        granularity = 'per_tensor'
        algorithm = 'minmax'

        from onnx import onnx_pb as onnx_proto
        for _, op in enumerate(self.quantizable_ops):
            if tune_cfg['op'][(op.name, op.op_type)]['activation']['dtype'] in \
                self.query_handler.get_fallback_list():
                quantize_config[op.name] = \
                    tune_cfg['op'][(op.name, op.op_type)]['activation']['dtype']
            else:
                node_config = copy.deepcopy(tune_cfg['op'][(op.name, op.op_type)])
                for tensor, config in tune_cfg['op'][(op.name, op.op_type)].items():
                    if 'granularity' not in config:
                        node_config[tensor]['granularity'] = granularity
                    if 'algorithm' not in config:
                        node_config[tensor]['algorithm'] = algorithm
                    if config['dtype'] == "int8":
                        node_config[tensor]['dtype'] = onnx_proto.TensorProto.INT8
                        if 'scheme' not in config:
                            node_config[tensor]['scheme'] = 'sym'
                    else:
                        node_config[tensor]['dtype'] = onnx_proto.TensorProto.UINT8
                        if 'scheme' not in config:
                            node_config[tensor]['scheme'] = 'asym'
                quantize_config[op.name] = node_config

        return quantize_config

    def _query_quantizable_ops(self, model):
        for node in model.graph.node:
            if node.op_type in self.quantizable_op_types and node not in self.quantizable_ops:
                self.quantizable_ops.append(node)

        return self.quantizable_ops

    def _query_quantizable_op_types(self):
        quantizable_op_types = self.query_handler.get_op_types_by_precision(precision='int8')
        return quantizable_op_types

    def evaluate(self, input_graph, dataloader, postprocess=None,
                 metrics=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        """The function is for evaluation if no given eval func

        Args:
            input_graph      : onnx model for evaluation
            dataloader       : dataloader for evaluation. neural_compressor.data.dataloader.ONNXDataLoader
            postprocess      : post-process for evalution. neural_compressor.data.transform.ONNXTransforms
            metrics:         : metrics for evaluation. neural_compressor.metric.ONNXMetrics
            measurer         : neural_compressor.objective.Measurer
            iteration(int)   : max iterations of evaluaton.
            tensorboard(bool): whether to use tensorboard for visualizaton
            fp32_baseline (boolen, optional): only for compare_label=False pipeline

        Returns:
            (float) evaluation results. acc, f1 e.g.
        """
        sess_options = ort.SessionOptions()
        if measurer:
            # https://github.com/microsoft/onnxruntime/issues/7347
            cores_per_instance = int(os.environ.get('CORES_PER_INSTANCE'))
            assert cores_per_instance > 0, "benchmark cores_per_instance should greater than 0"
            sess_options.intra_op_num_threads = cores_per_instance
        try:
            session = ort.InferenceSession(input_graph.model.SerializeToString(), sess_options)
        except:
            sess_options.register_custom_ops_library(ort_ext.get_library_path())
            session = ort.InferenceSession(input_graph.model.SerializeToString(), sess_options)
        results = []
        if metrics:
            for metric in metrics:
                metric.reset()
            self.fp32_preds_as_label = any([hasattr(metric, "compare_label") and \
                not metric.compare_label for metric in metrics]) 

        ort_inputs = {}
        len_inputs = len(session.get_inputs())
        inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]

        def eval_func(dataloader):
            for idx, (inputs, labels) in enumerate(dataloader):
                if not isinstance(labels, list):
                    labels = [labels]
                if len_inputs == 1:
                    ort_inputs.update(
                        inputs if isinstance(inputs, dict) else {inputs_names[0]: inputs}
                    )
                else:
                    assert len_inputs == len(inputs), \
                        'number of input tensors must align with graph inputs'

                    if isinstance(inputs, dict):  # pragma: no cover
                        ort_inputs.update(inputs)
                    else:
                        for i in range(len_inputs):
                            # in case dataloader contains non-array input
                            if not isinstance(inputs[i], np.ndarray):
                                ort_inputs.update({inputs_names[i]: np.array(inputs[i])})
                            else:
                                ort_inputs.update({inputs_names[i]: inputs[i]})

                if measurer is not None:
                    measurer.start()
                    predictions = session.run(None, ort_inputs)
                    measurer.end()
                else:
                    predictions = session.run(None, ort_inputs)

                if self.fp32_preds_as_label:
                    self.fp32_results.append(predictions) if fp32_baseline else \
                        results.append(predictions)

                if postprocess is not None:
                    predictions, labels = postprocess((predictions, labels))
                if metrics:
                    for metric in metrics:
                        if not hasattr(metric, "compare_label") or \
                            (hasattr(metric, "compare_label") and metric.compare_label):
                            metric.update(predictions, labels)
                if idx + 1 == iteration:
                    break

        if isinstance(dataloader, BaseDataLoader) and not self.benchmark:
            try:
                eval_func(dataloader)
            except Exception:  # pragma: no cover
                logger.warning(
                    "Fail to forward with batch size={}, set to {} now.".
                    format(dataloader.batch_size, 1))
                dataloader.batch(1)
                eval_func(dataloader)
        else:  # pragma: no cover
            eval_func(dataloader)

        if self.fp32_preds_as_label:
            from neural_compressor.adaptor.ox_utils.util import collate_preds
            if fp32_baseline:
                results = collate_preds(self.fp32_results)
                reference = results
            else:
                reference = collate_preds(self.fp32_results)
                results = collate_preds(results)
            for metric in metrics:
                if hasattr(metric, "compare_label") and not metric.compare_label:
                    metric.update(results, reference)

        acc = 0 if metrics is None else [metric.result() for metric in metrics]
        return acc if not isinstance(acc, list) or len(acc) > 1 else acc[0]

    def diagnosis_helper(self, fp32_model, int8_model, tune_cfg=None, save_path=None):
        from neural_compressor.utils.utility import dump_data_to_local
        from neural_compressor.adaptor.ox_utils.util import find_by_name
        if self.backend in ["qlinearops", "qoperator"]:
            supported_optype = ['Conv', 'MatMul', 'Concat', 'Attention', 'FusedConv',
                'Add', 'Mul', 'LeakyRelu', 'Sigmoid', 'GlobalAveragePool', 'AveragePool']
        elif self.backend == "qdq":
            supported_optype = ['Conv', 'MatMul', 'Concat', 'Attention', 'FusedConv',
                'LeakyRelu', 'Sigmoid', 'GlobalAveragePool', 'AveragePool']
        else:
            supported_optype = ['Conv', 'MatMul', 'Attention', 'LSTM']
        inspect_node_list = []
        int8_node_names = [i.name for i in int8_model.nodes()]
        for node in fp32_model.nodes():
            if node.op_type in supported_optype and node.name + '_quant' in int8_node_names:
                inspect_node_list.append(node.name)

        filtered_params = {}
        if self.min_max:
            for node_name in inspect_node_list:
                node = find_by_name(node_name, fp32_model.nodes())
                filtered_params[node_name] = {
                    'min': np.array(self.min_max[node.output[0]][0], dtype=np.float32),
                    'max': np.array(self.min_max[node.output[0]][1], dtype=np.float32)}
        if save_path:
            dump_data_to_local(filtered_params, save_path, 'dequan_min_max.pkl')
            dump_data_to_local(tune_cfg, save_path, 'cfg.pkl')
        return inspect_node_list, tune_cfg

    def save(self, model, path):
        """ save model

        Args:
            model (ModelProto): model to save
            path (str): save path
        """
        model.save(os.path.join(path, "best_model.onnx"))


@adaptor_registry
class ONNXRT_QLinearOpsAdaptor(ONNXRTAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        self.query_handler = ONNXRTQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "onnxrt_qlinear.yaml"))
        self.backend = "qlinearops"
        super().__init__(framework_specific_info)

@adaptor_registry
class ONNXRT_QOperatorAdaptor(ONNXRTAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        self.query_handler = ONNXRTQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "onnxrt_qlinear.yaml"))
        self.backend = "qlinearops"
        super().__init__(framework_specific_info)


@adaptor_registry
class ONNXRT_IntegerOpsAdaptor(ONNXRTAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        self.query_handler = ONNXRTQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "onnxrt_integer.yaml"))
        self.backend = "integerops"
        super().__init__(framework_specific_info)

@adaptor_registry
class ONNXRT_QDQAdaptor(ONNXRTAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        self.query_handler = ONNXRTQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "onnxrt_qdq.yaml"))
        self.backend = "qdq"
        super().__init__(framework_specific_info)

class ONNXRTQuery(QueryBackendCapability):

    def __init__(self, local_config_file=None):
        super().__init__()
        self.version = ort.__version__
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e: # pragma: no cover
                logger.info("Fail to parse {} due to {}.".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError("Please check if the format of {} follows Neural Compressor yaml schema.".
                                 format(self.cfg))

    def _get_specified_version_cfg(self, data): # pragma: no cover
        """Get the configuration for the current runtime.
        If there's no matched configuration in the input yaml, we'll
        use the `default` field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        from functools import cmp_to_key
        config = None

        def _compare(version1, version2):
            if Version(version1[0]) == Version(version2[0]):
                return 0
            elif Version(version1[0]) < Version(version2[0]):
                return -1
            else:
                return 1

        extended_cfgs = []
        for sub_data in data:
            if 'default' in sub_data['version']['name']:
                assert config == None, "Only one default config " \
                    "is allowed in framework yaml file."
                config = sub_data
            versions = sub_data['version']['name'] if \
                isinstance(sub_data['version']['name'], list) else \
                [sub_data['version']['name']]
            for version in versions:
                if version != 'default':
                    extended_cfgs.append((version, sub_data))

        extended_cfgs = sorted(extended_cfgs, key=cmp_to_key(_compare), reverse=True)
        for k, v in extended_cfgs:
            if Version(self.version) >= Version(k):
                config = v
                break

        return config

    def get_version(self): # pragma: no cover
        """Get the current backend version infomation.

        Returns:
            [string]: version string.
        """
        return self.cur_config['version']['name']

    def get_precisions(self): # pragma: no cover
        """Get supported precisions for current backend.

        Returns:
            [string list]: the precisions' name.
        """
        return [i.strip() for i in self.cur_config['precisions']['names'].split(',')]

    def get_op_types(self): # pragma: no cover
        """Get the supported op types by all precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the op types.
        """
        return self.cur_config['ops']

    def get_quantization_capability(self):
        """Get the supported op types' quantization capability.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        return self.cur_config['capabilities']

    def get_op_types_by_precision(self, precision):
        """Get op types per precision

        Args:
            precision (string): precision name

        Returns:
            [string list]: A list composed of op type.
        """
        #assert precision in list(self.cur_config['ops'].keys())
        if precision in list(self.cur_config['ops'].keys()):
            return self.cur_config['ops'][precision]
        else:
            return []

    def get_graph_optimization(self):
        """ Get onnxruntime graph optimization level"""
        optimization_levels = {'DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                               'ENABLE_BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                               'ENABLE_EXTENDED': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                               'ENABLE_ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL}
 
        level = self.cur_config['graph_optimization']['level']
        assert level in optimization_levels, "the optimization choices \
                                              are {}".format(optimization_levels.keys())
        return optimization_levels[level]

    def get_fallback_list(self):
        return list(self.cur_config['ops'].keys() - self.cur_config['capabilities'].keys())
