# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Hrida-Hybrid model (ZLabs Innovation).

Hrida-Hybrid checkpoints store Qwen3.5-MoE text weights under clean
``model.*`` names plus an all-zero router selection bias. With the optional
Hrida extensions off, the model is computed by vLLM's Qwen3.5-MoE text model.
"""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.models.qwen3_5 import Qwen3_5MoeForCausalLM

_ROUTER_BIAS = "e_score_correction_bias"


def _unsupported_features(config) -> list[str]:
    problems = []
    if getattr(config, "n_domain_experts", 0):
        problems.append(f"{config.n_domain_experts} domain experts per layer")
    layer_types = set(getattr(config, "layer_types", None) or [])
    extra = layer_types - {"linear_attention", "full_attention"}
    if extra:
        problems.append(f"attention layer types {sorted(extra)}")
    if (getattr(config, "hc_mult", None) or 1) > 1:
        problems.append("hyper-connections (hc_mult > 1)")
    if getattr(config, "compressed_attention", None):
        problems.append("compressed attention")
    return problems


class HridaHybridForCausalLM(Qwen3_5MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        problems = _unsupported_features(config)
        if problems:
            raise ValueError(
                "Hrida-Hybrid features not supported by vLLM yet: " + "; ".join(problems)
            )
        # The Qwen3.5 decoder layers select their MLP by model_type.
        config.model_type = "qwen3_5_moe_text"
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _drop_router_bias():
            for name, tensor in weights:
                if name.endswith(_ROUTER_BIAS):
                    if torch.count_nonzero(tensor) != 0:
                        raise ValueError(
                            f"{name} is non-zero; Qwen3.5-MoE routing has no "
                            "selection bias"
                        )
                    continue
                yield name, tensor

        return super().load_weights(_drop_router_bias())
