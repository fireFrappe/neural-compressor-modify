# -*- coding: utf-8 -*-
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
"""Configuration generator class."""
import json
import os
from typing import Any

from neural_compressor.ux.components.config_generator.config_generator import ConfigGenerator


class ProfilingConfigGenerator(ConfigGenerator):
    """Configuration generator class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize configuration generator."""
        super().__init__(*args, **kwargs)
        data = kwargs.get("data", {})
        self.num_threads: int = data["num_threads"]

    def generate(self) -> None:
        """Dump profiling configuration to json file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as json_file:
            json.dump(self.serialize(), json_file, indent=4)
