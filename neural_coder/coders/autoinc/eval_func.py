# Copyright (c) 2022 Intel Corporation
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

from ... import globals
from . import domain
from ...utils.line_operation import get_line_indent_level
class Eval_Func(object):
    def __init__(self):
        pass
    def register_transformation(self):
        domain_ = domain.determine_domain(globals.list_code_path[0])
        if domain_ == 'transformers_trainer':
            lines = [
                'trainer.model = model', 
                'metrics = trainer.evaluate() # check if all tasks do not have parameters in evaluate()', 
                'keys = [', 
                '    "eval_accuracy",', 
                '    "eval_bleu",', 
                '    "eval_matthews_correlation",', 
                '    "eval_pearsonr",', 
                '    "eval_precision",', 
                '    "eval_recall",', 
                '    "eval_rouge",', 
                '    "eval_sacrebleu",', 
                '    "eval_spearmanr",', 
                '    "eval_mcc",', 
                '    "eval_acc",', 
                '    "eval_acc_and_f1",', 
                '    "eval_corr",', 
                '    "eval_mnli/acc",', 
                '    "eval_mnli-mm/acc",', 
                '] # METRIC_TAGS in transformers', 
                'for key in keys:', 
                '    if key in metrics.keys():', 
                '        return metrics[key]', 
                'assert False, \"No metric returned, Please check inference metric!\"'
                ]
            for index, line in enumerate(lines):
                if index != 0:
                    lines[index] = '[+] ' + ' ' * 8 + line
            lines = '\n'.join(lines)
            globals.list_eval_func_lines.append(lines)
        elif domain_ == 'transformers_no_trainer':
            pass
        elif domain_ == 'torchvision':
            # search for 'validate()'
            codes = open(globals.list_code_path[0], 'r').read().split('\n')
            lines = []
            for index, line in enumerate(codes):
                if 'def validate(' in line:
                    start = index
                    start_indent = get_line_indent_level(codes[start])
                    for i in range(start+1, len(codes)):
                        if codes[i] == '':
                            continue
                        line_indent = get_line_indent_level(codes[i])
                        if line_indent > start_indent:
                            change_indent = line_indent - 4
                            lines.append(' ' * change_indent + codes[i].lstrip())
                        # no 'print'
                        else:
                            break
                    break
                else:
                    pass
            for index, line in enumerate(lines):
                if 'return' in line:
                    indent = get_line_indent_level(line)
                    line_list = line.split()
                    line_list[1] = 'float(' + line_list[1] + ')'
                    lines[index] = ' ' * indent + ' '.join(line_list)
            for index, line in enumerate(lines):
                if index != 0:
                    lines[index] = '[+] ' + ' ' * 8 + line
            lines = '\n'.join(lines)
            globals.list_eval_func_lines.append(lines)
        else: # random model
            pass
