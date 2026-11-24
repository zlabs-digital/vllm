# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config for Hrida-Hybrid (model_type ``hrida_hybrid``) by ZLabs Innovation.

Hrida-Hybrid is built from Qwen3.5-MoE weights. With its optional extensions
off (no domain experts, no hyper-connections, only Gated DeltaNet / full
attention layers and MoE MLPs) it computes exactly the Qwen3.5-MoE text model,
so this config only renames fields to the names vLLM's Qwen3.5 code reads.
"""

from transformers import Qwen3_5MoeTextConfig


class HridaHybridConfig(Qwen3_5MoeTextConfig):
    model_type = "hrida_hybrid"

    def __init__(self, n_routed_experts=None, mlp_layer_types=None, **kwargs):
        if n_routed_experts is not None:
            kwargs.setdefault("num_experts", n_routed_experts)
        if mlp_layer_types:
            unknown = set(mlp_layer_types) - {"moe", "dense"}
            if unknown:
                raise ValueError(
                    f"Hrida-Hybrid MLP layer types {sorted(unknown)} are not supported"
                )
            # An all-MoE stack is the Qwen3.5 default; only mixed stacks are kept.
            if set(mlp_layer_types) != {"moe"}:
                kwargs["mlp_layer_types"] = [
                    "sparse" if t == "moe" else t for t in mlp_layer_types
                ]
        kwargs.setdefault("attn_output_gate", True)
        kwargs.setdefault("mamba_ssm_dtype", "float32")
        kwargs.setdefault("mlp_only_layers", [])
        kwargs.setdefault("mtp_use_dedicated_embeddings", False)
        super().__init__(**kwargs)
        self.n_routed_experts = self.num_experts
