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
from typing import Any

from verl.single_controller.ray import RayWorkerGroup


def update_weights(
    actor_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, swap_ema: bool = False
):
    """Update weights from actor worker group to rollout worker group.

    Args:
        actor_wg (RayWorkerGroup): The actor worker group.
        rollout_wg (RayWorkerGroup): The rollout worker group.
        swap_ema (bool): Whether to swap EMA weights. Default is False.
    """
    if actor_wg is rollout_wg:
        return

    params_with_config: tuple[dict[str, Any], ...] = actor_wg.get_params(
        swap_ema=swap_ema
    )
    # all workers are expected to have the same parameters and configs under DP/FSDP.
    # therefore, we use the parameters from rank 0, and distribute to all rollout workers.
    rollout_wg.update_weights(params_with_config[0])
