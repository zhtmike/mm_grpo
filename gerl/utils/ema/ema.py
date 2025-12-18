# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from typing import Iterable, Optional

import torch


class EMAModuleWrapper:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        device: str = "cpu",
    ):
        super().__init__()
        self.ema_parameters = [p.detach().to(device) for p in parameters]
        self.temp_stored_parameters: Optional[list[torch.Tensor]] = None
        self.decay = decay
        self.device = device

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        one_minus_decay = 1 - self.decay

        for ema_parameter, parameter in zip(self.ema_parameters, parameters):
            if parameter.requires_grad:
                ema_parameter.add_(
                    one_minus_decay
                    * (parameter.to(ema_parameter.device) - ema_parameter)
                )

    @torch.no_grad()
    def copy_ema_to_model(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy the EMA parameters to the model parameters.
        """
        self.temp_stored_parameters = [p.detach().cpu() for p in parameters]
        for ema_parameter, parameter in zip(self.ema_parameters, parameters):
            parameter.copy_(ema_parameter)

    @torch.no_grad()
    def copy_temp_to_model(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy the temporary stored parameters back to the model parameters.
        """
        if self.temp_stored_parameters is None:
            raise RuntimeError(
                "No temporary parameters stored. Call copy_ema_to_model first."
            )
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters):
            parameter.copy_(temp_parameter)

        self.temp_stored_parameters = None
