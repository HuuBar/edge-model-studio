import os
import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from huggingface_hub import hf_hub_download
try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from configs import EConfig
    from utils_c import *
    from choices import *


_INV_FREQ_CACHE = {}


def _get_inv_freq(
        dim: int,
        base: float,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = torch.device(device) if device is not None else torch.device("cpu")
    key = (int(dim), float(base), str(device))

    cached = _INV_FREQ_CACHE.get(key)
    if cached is not None:
        return cached

    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device).float() / dim)
    )
    _INV_FREQ_CACHE[key] = inv_freq
    return inv_freq


def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full(
        (tgt_len, tgt_len),
        torch.finfo(dtype).min,
        device=device,
        dtype=dtype,
    )
    mask = torch.triu(mask, diagonal=1)

    if past_key_values_length > 0:
        past_mask = torch.zeros(
            tgt_len,
            past_key_values_length,
            dtype=dtype,
            device=device,
        )
        mask = torch.cat([past_mask, mask], dim=-1)

    return mask[None, None, :, :].expand(
        bsz,
        1,
        tgt_len,
        tgt_len + past_key_values_length,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = _get_inv_freq(self.dim, self.base, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(dtype=dtype)
        sin_cached = emb.sin().to(dtype=dtype)
        self.register_buffer("cos_cached", cos_cached[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", sin_cached[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(dtype=dtype)
        sin_cached = emb.sin().to(dtype=dtype)
        self.register_buffer("cos_cached", cos_cached[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", sin_cached[None, None, :, :], persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = _get_inv_freq(self.dim, base, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(dtype=dtype)
        sin_cached = emb.sin().to(dtype=dtype)
        self.register_buffer("cos_cached", cos_cached[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", sin_cached[None, None, :, :], persistent=False)


class LlamaAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _project_qkv(
            self,
            hidden_states: torch.Tensor,
            bsz: int,
            q_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states into query, key and value states."""
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp,
                dim=0,
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            query_states = torch.cat(
                [F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            key_states = torch.cat(
                [F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            value_states = torch.cat(
                [F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        return query_states, key_states, value_states


    def _apply_rope_and_cache(
            self,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            position_ids: Optional[torch.LongTensor],
            past_key_value: Optional[Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Apply RoPE to Q/K and append cached K/V for decoding."""
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        return query_states, key_states, value_states, kv_seq_len

    def _compute_attention(
            self,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            bsz: int,
            q_len: int,
            kv_seq_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute scaled dot-product attention with optional attention mask."""
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        expected_attn_shape = (bsz, self.num_heads, q_len, kv_seq_len)
        if attn_weights.size() != expected_attn_shape:
            raise ValueError(
                f"Attention weights should be of size {expected_attn_shape}, "
                f"but is {attn_weights.size()}"
            )
        if attention_mask is not None:
            expected_mask_shape = (bsz, 1, q_len, kv_seq_len)
            if attention_mask.size() != expected_mask_shape:
                raise ValueError(
                    f"Attention mask should be of size {expected_mask_shape}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        expected_output_shape = (bsz, self.num_heads, q_len, self.head_dim)
        if attn_output.size() != expected_output_shape:
            raise ValueError(
                f"`attn_output` should be of size {expected_output_shape}, "
                f"but is {attn_output.size()}"
            )
        return attn_output, attn_weights

    def _project_output(
            self,
            attn_output: torch.Tensor,
            bsz: int,
            q_len: int,
    ) -> torch.Tensor:
        """Merge attention heads and apply output projection."""
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp,
                dim=2,
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp,
                dim=1,
            )
            return sum(
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            )
        return self.o_proj(attn_output)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Run one LLaMA attention layer.
        Steps:
        1. Project hidden states to Q/K/V.
        2. Apply RoPE and append cached K/V if provided.
        3. Compute scaled dot-product attention.
        4. Project attention output back to hidden size.
        """
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self._project_qkv(hidden_states, bsz, q_len)
        query_states, key_states, value_states, kv_seq_len = self._apply_rope_and_cache(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        present_key_value = (key_states, value_states) if use_cache else None
        attn_output, attn_weights = self._compute_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            bsz=bsz,
            q_len=q_len,
            kv_seq_len=kv_seq_len,
        )
        attn_output = self._project_output(attn_output, bsz, q_len)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, present_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_emb: torch.Tensor,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def len_list(x, n):
    return [i for i in x if len(i) <= n]


def _is_valid_choice_list(value):
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, (list, tuple)) and len(x) > 0 for x in value)
    )


def _fallback_static_tree_choices(total_tokens: int, max_depth: int, top_k: int):
    """
    Generate fallback tree paths by depth.
    Example:
        total_tokens=5, max_depth=3, top_k=2
        -> [[0], [1], [0, 0], [0, 1], [1, 0]]
    Args:
        total_tokens: max number of paths.
        max_depth: max path length.
        top_k: branch count per level.
    """
    choices = []
    frontier = [()]
    for _depth in range(1, max_depth + 1):
        next_frontier = []
        for prefix in frontier:
            for rank in range(top_k):
                path = prefix + (rank,)
                choices.append(list(path))
                next_frontier.append(path)
                if len(choices) >= total_tokens:
                    return choices
        frontier = next_frontier
    return choices


def _normalize_static_tree_choices(
    raw_choices,
    total_tokens: int,
    max_depth: int,
    top_k: int,
    fill_to_total_tokens: bool = False,
):
    expanded = set()
    if raw_choices is not None:
        for path in raw_choices:
            if not isinstance(path, (list, tuple)):
                continue
            if len(path) == 0 or len(path) > max_depth:
                continue
            try:
                path = tuple(int(x) for x in path)
            except Exception:
                continue
            if any(x < 0 or x >= top_k for x in path):
                continue
            for i in range(1, len(path) + 1):
                expanded.add(path[:i])
    if fill_to_total_tokens and len(expanded) < total_tokens:
        for path in _fallback_static_tree_choices(total_tokens, max_depth, top_k):
            key = tuple(path)
            if key not in expanded:
                expanded.add(key)
            if len(expanded) >= total_tokens:
                break
    choices = sorted(expanded, key=lambda x: (len(x), x))
    choices = [list(x) for x in choices]
    if fill_to_total_tokens:
        choices = choices[:total_tokens]
    return choices


def _pick_default_tree_choices(total_tokens: int, max_depth: int, top_k: int):
    preferred_names = [
        "mc_sim_7b_63",
    ]
    raw = None
    picked_name = None
    for name in preferred_names:
        value = globals().get(name)
        if _is_valid_choice_list(value):
            raw = value
            picked_name = name
            break
    if raw is None:
        raise ValueError(
            "No configured static tree found. Expected choices.py to define "
            "mc_sim_7b_63 as a non-empty list of non-empty paths."
        )
    choices = _normalize_static_tree_choices(
        raw,
        total_tokens=total_tokens,
        max_depth=max_depth,
        top_k=top_k,
        fill_to_total_tokens=False,
    )
    return choices


def _choices_to_parent_rank(sorted_choices):
    path_to_node = {(): 0}
    parent = []
    rank = []
    depth_to_nodes = []
    for node_id, path in enumerate(sorted_choices, start=1):
        path = tuple(path)
        # Empty path means root node, which is already represented by () -> 0.
        # Skip it to avoid path[-1] IndexError.
        if not path:
            continue
        parent_path = path[:-1]
        if parent_path not in path_to_node:
            raise ValueError(
                f"Static tree choices are not prefix-closed. "
                f"Missing parent of {path}."
            )
        p = path_to_node[parent_path]
        r = int(path[-1])
        path_to_node[path] = node_id
        parent.append(p)
        rank.append(r)
        depth = len(path)
        missing = depth - len(depth_to_nodes)
        if missing > 0:
            # Use list comprehension instead of [[]] * missing to avoid
            # multiple entries sharing the same list object.
            depth_to_nodes.extend([[] for _ in range(missing)])
        depth_to_nodes[depth - 1].append(node_id)
    return parent, rank, depth_to_nodes


def _build_final_static_tree_tensors(parent, device, sort_retrieve: bool = False):
    total_tokens = len(parent)
    tree_mask = torch.eye(total_tokens + 1, dtype=torch.bool, device=device)
    tree_mask[:, 0] = True
    for node in range(1, total_tokens + 1):
        p = parent[node - 1]
        if p > 0:
            tree_mask[node] |= tree_mask[p]
    tree_position_ids = tree_mask.sum(dim=1).long() - 1
    non_leaf = set(parent)
    leaves = [node for node in range(1, total_tokens + 1) if node not in non_leaf]
    if not leaves:
        leaves = [total_tokens]
    max_depth = int(tree_position_ids[leaves].max().item()) + 1
    retrieve_indices = torch.full(
        (len(leaves), max_depth), -1, dtype=torch.long, device=device
    )
    for row, leaf in enumerate(leaves):
        path = []
        cur = leaf
        while True:
            path.append(cur)
            if cur == 0:
                break
            cur = parent[cur - 1]
        path.reverse()
        retrieve_indices[row, :len(path)] = torch.tensor(path, dtype=torch.long, device=device)
    if sort_retrieve:
        maxitem = total_tokens + 1
        rows = retrieve_indices.detach().cpu().tolist()

        def custom_sort(lst):
            return [x if x >= 0 else maxitem for x in lst]
        order = sorted(range(len(rows)), key=lambda i: custom_sort(rows[i]))
        retrieve_indices = retrieve_indices[torch.tensor(order, dtype=torch.long, device=device)]
    return tree_mask.float()[None, None], tree_position_ids, retrieve_indices


class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        if load_emb and not hasattr(config, "target_hidden_size"):
            from safetensors import safe_open
            import json
            try:
                index_json_path = os.path.join(path, "model.safetensors.index.json")
                if not os.path.exists(index_json_path):
                    index_json_path = hf_hub_download(path, "model.safetensors.index.json")
                with open(index_json_path, "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                local_emb_path = os.path.join(path, emb_path)
                if not os.path.exists(local_emb_path):
                    local_emb_path = hf_hub_download(path, emb_path)
                with safe_open(local_emb_path,
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except Exception:
                index_json_path = os.path.join(path, "pytorch_model.bin.index.json")
                if not os.path.exists(index_json_path):
                    index_json_path = hf_hub_download(path, "pytorch_model.bin.index.json")
                with open(index_json_path, "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                local_emb_path = os.path.join(path, emb_path)
                if not os.path.exists(local_emb_path):
                    local_emb_path = hf_hub_download(path, emb_path)
                weights = torch.load(local_emb_path)
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor
        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        self.hidden_size = config.hidden_size
        self.midlayer = LlamaDecoderLayeremb(config)
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)
        else:
            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        d2t = torch.zeros((config.draft_vocab_size), dtype=torch.long)
        t2d = torch.zeros((config.vocab_size), dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)
        self.static_tree_choices = None
        self.static_parent = None
        self.static_rank = None
        self.static_depth_to_nodes = None
        self.static_tree_mask = None
        self.static_tree_position_ids = None
        self.static_retrieve_indices = None
        self.static_retrieve_indices_sorted = None
        self.static_tree_device = None
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def _draft_to_target(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        if self.config.vocab_size == self.config.draft_vocab_size:
            return draft_token_ids
        return draft_token_ids + self.d2t[draft_token_ids]

    def _rebuild_static_tree(self, device):
        max_depth = max(1, self.depth + 1)
        self.static_tree_choices = _pick_default_tree_choices(
            total_tokens=self.total_tokens,
            max_depth=max_depth,
            top_k=self.top_k,
        )
        if len(self.static_tree_choices) == 0:
            raise ValueError(
                "No valid static tree choices found. Check choices.py and make sure "
                "all ranks are in [0, self.top_k)."
            )
        if self.total_tokens != len(self.static_tree_choices):
            self.total_tokens = len(self.static_tree_choices)
        self.static_parent, self.static_rank, self.static_depth_to_nodes = _choices_to_parent_rank(
            self.static_tree_choices
        )
        self.static_tree_mask, self.static_tree_position_ids, self.static_retrieve_indices = (
            _build_final_static_tree_tensors(self.static_parent, device=device, sort_retrieve=False)
        )
        _, _, self.static_retrieve_indices_sorted = _build_final_static_tree_tensors(
            self.static_parent, device=device, sort_retrieve=True
        )
        self.static_tree_device = device

    def _ensure_static_tree(self, device):
        if (
            self.static_parent is None
            or self.static_rank is None
            or self.static_depth_to_nodes is None
            or self.static_tree_device != device
        ):
            self._rebuild_static_tree(device)

    def _make_static_step_tree_mask(self, current_nodes, device):
        if len(current_nodes) == 0:
            raise ValueError("current_nodes must not be empty")
        first_node = current_nodes[0]
        prev_count = first_node - 1
        cur_count = len(current_nodes)
        current_pos = {node: i for i, node in enumerate(current_nodes)}
        mask = torch.zeros((cur_count, prev_count + cur_count), dtype=torch.bool, device=device)
        for row, node in enumerate(current_nodes):
            mask[row, prev_count + row] = True
            parent = self.static_parent[node - 1]
            while parent > 0:
                if parent in current_pos:
                    mask[row, prev_count + current_pos[parent]] = True
                else:
                    mask[row, parent - 1] = True
                parent = self.static_parent[parent - 1]
        return mask.float()[None, None]

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)
        self._rebuild_static_tree(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask.to(combined_attention_mask.device)
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min
        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        past_key_value = past_key_values[0] if past_key_values is not None else None
        layer_outputs = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=True,
        )
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        hidden_states = layer_outputs[0]
        if use_cache:
            return hidden_states, next_decoder_cache
        return hidden_states, None

    def reset_kv(self):
        self.stable_kv = None

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        device = hidden_states.device
        self._ensure_static_tree(device)
        input_ids = input_ids.to(device)
        sample_token = input_ids[:, -1]
        input_ids = input_ids[:, 1:].to(device)
        len_posi = input_ids.shape[1]
        self.reset()
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids[:, kv_len:],
                past_key_values=self.stable_kv,
                use_cache=True,
            )
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        root_hidden = out_hidden[:, -1]  # [1, hidden]
        total_tokens = self.total_tokens
        hidden_by_node = [None for _ in range(total_tokens + 1)]
        token_by_node = [None for _ in range(total_tokens + 1)]
        hidden_by_node[0] = root_hidden[0]
        for depth_idx, current_nodes in enumerate(self.static_depth_to_nodes, start=1):
            if len(current_nodes) == 0:
                continue
            parent_nodes = [self.static_parent[node - 1] for node in current_nodes]
            parent_hidden = torch.stack([hidden_by_node[p] for p in parent_nodes], dim=0)  # [num_nodes, hidden]
            logits = self.lm_head(self.norm(parent_hidden))
            log_probs = self.logsoftmax(logits)
            top = torch.topk(log_probs, self.top_k, dim=-1)
            topk_index = top.indices
            rows = torch.arange(len(current_nodes), device=device)
            ranks = torch.tensor(
                [self.static_rank[node - 1] for node in current_nodes],
                dtype=torch.long,
                device=device,
            )
            draft_vocab_tokens = topk_index[rows, ranks]
            target_vocab_tokens = self._draft_to_target(draft_vocab_tokens)
            for node, token in zip(current_nodes, target_vocab_tokens):
                token_by_node[node] = token
            self.tree_mask = self._make_static_step_tree_mask(current_nodes, device)
            position_ids = torch.full(
                (len(current_nodes),),
                len_posi + depth_idx - 1,
                dtype=torch.long,
                device=device,
            )
            out_hidden, past_key_values = self(
                parent_hidden[None],
                input_ids=target_vocab_tokens[None],
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            for row, node in enumerate(current_nodes):
                hidden_by_node[node] = out_hidden[0, row]
        self.reset()
        draft_tokens = torch.empty((total_tokens + 1,), dtype=torch.long, device=device)
        draft_tokens[0] = sample_token[0]
        for node in range(1, total_tokens + 1):
            if token_by_node[node] is None:
                raise RuntimeError(f"Static draft tree did not generate node {node}.")
            draft_tokens[node] = token_by_node[node]
        draft_tokens = draft_tokens[None]
        tree_mask = self.static_tree_mask.to(device)
        tree_position_ids = self.static_tree_position_ids.to(device)
        if logits_processor is not None:
            retrieve_indices = self.static_retrieve_indices_sorted.to(device)
        else:
            retrieve_indices = self.static_retrieve_indices.to(device)
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    print(model)
