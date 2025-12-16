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
import logging
import os
from operator import attrgetter
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_negative_prompt_embedding(
    pipeline: "DiffusionPipeline",
    negative_prompt: str = "",
    max_sequence_length: int = 128,
) -> dict[str, torch.FloatTensor]:
    from diffusers import StableDiffusion3Pipeline

    if isinstance(pipeline, StableDiffusion3Pipeline):
        prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            negative_prompt,
            None,
            None,
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )
        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_pooled_prompt_embeds": pooled_prompt_embeds,
        }
    else:
        raise NotImplementedError(f"Pipeline type {type(pipeline)} not supported yet.")


def load_peft_weight_from_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    adapter_name: str = "default",
    parameter_prefix: str = "lora_",
):
    """Load peft weights from state_dict into model, supports torch compiling.
    Modified from https://github.com/huggingface/peft/blob/e70542269c4f9668dbf9607ca383333ef7723b48/src/peft/utils/hotswap.py#L369
    """
    state_dict = _map_state_dict_for_hotswap(state_dict, adapter_name, parameter_prefix)

    is_compiled = hasattr(model, "_orig_mod")
    missing_keys = {
        k for k in model.state_dict() if (parameter_prefix in k) and (adapter_name in k)
    }
    unexpected_keys = []

    for key, new_val in state_dict.items():
        try:
            old_val = attrgetter(key)(model)
        except AttributeError:
            unexpected_keys.append(key)
            continue

        if is_compiled:
            missing_keys.remove("_orig_mod." + key)
        else:
            missing_keys.remove(key)

    if unexpected_keys:
        msg = f"Loading the peft adapter did not succeed, unexpected keys: {unexpected_keys}."
        raise RuntimeError(msg)

    if missing_keys:
        logger.warning(f"Missing keys when loading the peft adapter: {missing_keys}.")

    # actual swapping
    for key, new_val in state_dict.items():
        # swap actual weights
        # no need to account for potential _orig_mod in key here, as torch handles that
        old_val = attrgetter(key)(model)
        new_val = new_val.to(old_val.data.device)

        if not is_compiled:
            try:
                torch.utils.swap_tensors(old_val, new_val)
                continue
            except RuntimeError:
                is_compiled = True

        # Compiled models don't work with swap_tensors because there are weakrefs for the tensor. It is unclear if
        # this workaround could not cause trouble but the tests indicate that it works.
        if old_val.shape == new_val.shape:
            old_val.data.copy_(new_val.data)
        else:
            raise ValueError(
                f"Cannot load adapter weights, shape mismatch for {key}: {old_val.shape} vs {new_val.shape}. Please "
                "make sure that the adapter was trained with the same model and the same LoRA rank."
            )


def _map_state_dict_for_hotswap(sd, adapter_name, parameter_prefix):
    # copied from https://github.com/huggingface/diffusers/blob/a748a839add5fe9f45a66e45dd93d8db0b45ce0f/src/diffusers/loaders/peft.py#L311
    new_sd = {}
    for k, v in sd.items():
        if k.endswith(parameter_prefix + "A.weight") or k.endswith(
            parameter_prefix + "B.weight"
        ):
            k = k[: -len(".weight")] + f".{adapter_name}.weight"
        elif k.endswith(parameter_prefix + "B.bias"):  # lora_bias=True option
            k = k[: -len(".bias")] + f".{adapter_name}.bias"
        new_sd[k] = v
    return new_sd
