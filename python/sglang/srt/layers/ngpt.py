from typing import Optional, Tuple

import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.utils import is_cpu, set_weight_attrs

_is_cpu = is_cpu()


@torch.compile
def l2norm(x: torch.Tensor, layernorm_epsilon) -> torch.Tensor:
    x_float = x.float()
    res = x_float / (x_float.norm(p=2, dim=-1, keepdim=True) + layernorm_epsilon)
    return res.type_as(x)


@torch.compile
def l2normv2(x: torch.Tensor, layernorm_epsilon):
    x_float = x.float()
    if is_dp_attention_enabled():
        tp_size = get_attention_tp_size()
        if tp_size > 1:
            local_norm = x_float.square().sum(dim=-1, keepdim=True)
            weight_norm_square = get_attention_tp_group().all_reduce(local_norm)
            global_norm = weight_norm_square.sqrt()
            res = x_float / (global_norm + layernorm_epsilon)
        else:
            norm = x_float.norm(p=2, dim=-1, keepdim=True)
            res = x_float / (norm + layernorm_epsilon)
    else:
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            local_norm = x_float.square().sum(dim=-1, keepdim=True)
            weight_norm_square = tensor_model_parallel_all_reduce(local_norm)
            global_norm = weight_norm_square.sqrt()
            res = x_float / (global_norm + layernorm_epsilon)
        else:
            norm = x_float.norm(p=2, dim=-1, keepdim=True)
            res = x_float / (norm + layernorm_epsilon)
    return res.type_as(x)


@torch.compile
def apply_hyper_spherical_fusion(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    layernorm_epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_states = l2norm(hidden_states, layernorm_epsilon) * alpha
    residual = l2norm(residual, layernorm_epsilon) * (1 - alpha)
    return hidden_states, residual


@torch.compile
def scale_operation(
    tensor: torch.Tensor, scaling_up_factor: torch.Tensor, layernorm_epsilon: float
) -> torch.Tensor:
    split_dim = tensor.size(-1) // 2
    up_slice = tensor[..., split_dim:]
    up_slice = scaling_up_factor * l2normv2(up_slice, layernorm_epsilon)
    tensor[..., split_dim:] = up_slice
    return tensor


class ScaleUpNorm(torch.nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        layernorm_epsilon: float = 1e-6,
        params_dtype: Optional[torch.dtype] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
    ):
        super().__init__()

        self.use_presharded_weights = use_presharded_weights
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size

        self.origin_intermediate_size = intermediate_size
        self.intermediate_size = divide(intermediate_size, tp_size)
        self.layernorm_epsilon = layernorm_epsilon

        self.weight = Parameter(
            torch.empty(
                self.intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):

        param_data = param.data

        shard_size = self.intermediate_size
        start_idx = self.tp_rank * shard_size

        if _is_cpu:
            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data,
                loaded_weight,
                0,  # param_data_start
                start_idx,
                0,  # shard dim
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            if not self.use_presharded_weights:
                loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        loaded_weight = loaded_weight * (self.origin_intermediate_size**0.5)
        param_data.copy_(loaded_weight)

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = scale_operation(x, self.weight, self.layernorm_epsilon)
        return x
