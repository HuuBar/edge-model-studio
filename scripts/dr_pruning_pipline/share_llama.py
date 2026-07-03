import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

# ==========================================
# 1. 配置类
# ==========================================
@dataclass
class SharedLlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32      # 逻辑上的层数
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

# ==========================================
# 2. 基础组件：RMSNorm & RoPE
# ==========================================
class LlamaRMSNorm(nn.Module):
    """均方根层归一化 (Root Mean Square Layer Normalization)"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算旋转位置编码的频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """调整维度以支持广播"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """应用旋转位置编码 (RoPE)"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ==========================================
# 3. 注意力机制与 MLP
# ==========================================
class LlamaAttention(nn.Module):
    """多头/分组查询注意力机制 (GQA / MQA 支持)"""
    def __init__(self, config: SharedLlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        xq = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        xk = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        xv = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 处理 KV Cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=1)
            xv = torch.cat([v_cache, xv], dim=1)
        
        new_kv_cache = (xk, xv) if self.config.use_cache else None

        # GQA/MQA 广播
        if self.num_key_value_groups > 1:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2)

        xq = xq.transpose(1, 2) # (bsz, num_heads, q_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(output), new_kv_cache

class LlamaMLP(nn.Module):
    """基于 SwiGLU 的多层感知机"""
    def __init__(self, config: SharedLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# ==========================================
# 4. Decoder 层定义
# ==========================================
class LlamaDecoderLayer(nn.Module):
    """标准 Llama 解码器层"""
    def __init__(self, config: SharedLlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # Attention 模块残差连接
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv_cache = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # MLP 模块残差连接
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache

# ==========================================
# 5. Shared Llama 核心架构
# ==========================================
class SharedLlamaModel(nn.Module):
    """
    共享权重的 Llama 模型。
    它只实例化一个 (或极少数几个) LlamaDecoderLayer，
    但在前向传播中循环调用多次以达到 num_hidden_layers 的逻辑深度。
    """
    def __init__(self, config: SharedLlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # 核心：整个模型只有一个 Decoder Layer 的物理权重！
        self.shared_layer = LlamaDecoderLayer(config)
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 预计算 RoPE 频率
        freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            config.max_position_embeddings * 2
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        bsz, seq_len = input_shape
        if seq_len <= 1:
            return None
        
        mask = torch.full((seq_len, seq_len), float("-inf"), device=inputs_embeds.device)
        mask = torch.triu(mask, diagonal=1)
        
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(seq_len, past_key_values_length, device=mask.device), mask], dim=-1)
            
        return mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len + past_key_values_length)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        bsz, seq_len = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        
        freqs_cis = self.freqs_cis[past_key_values_length : past_key_values_length + seq_len]

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (bsz, seq_len), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        next_decoder_cache = () if self.config.use_cache else None

        # 循环调用同一个 shared_layer num_hidden_layers 次
        for idx in range(self.config.num_hidden_layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            
            hidden_states, kv_cache = self.shared_layer(
                hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                kv_cache=layer_past,
            )
            
            if self.config.use_cache:
                next_decoder_cache += (kv_cache,)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache

class SharedLlamaForCausalLM(nn.Module):
    def __init__(self, config: SharedLlamaConfig):
        super().__init__()
        self.config = config
        self.model = SharedLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values
        )
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return logits, loss, past_key_values

    # ==========================================
    # 6. 生成工具：Top-K / Top-P 采样
    # ==========================================
    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.LongTensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: int = 50, 
        top_p: float = 0.9,
    ):
        self.eval()
        bsz = input_ids.shape[0]
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # 只有第一次传递完整的序列，后续只传递最后一个 token 并利用 cache
            curr_input = input_ids if past_key_values is None else input_ids[:, -1:]
            
            logits, _, past_key_values = self(
                input_ids=curr_input, 
                past_key_values=past_key_values
            )
            
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-K 过滤
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            # Top-P (Nucleus) 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 遇到 EOS token 即停止生成
            if (next_token == self.config.eos_token_id).all():
                break
                
        return input_ids

# ==========================================
# 7. 测试与运行示例
# ==========================================
if __name__ == "__main__":
    # 使用较小的配置进行快速测试
    config = SharedLlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=6,  # 逻辑层数是6
        num_attention_heads=8,
        max_position_embeddings=512
    )
    
    # 初始化 Shared Llama
    print("正在初始化 Shared Llama 模型...")
    model = SharedLlamaForCausalLM(config)
    
    # 打印参数量对比
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    print(f"总参数量: {count_parameters(model):,}")
    print("注意：因为我们复用了 Decoder Layer 6 次，实际参数量远小于标准 6 层 Llama。")
    
    # 模拟输入
    dummy_input_ids = torch.randint(0, config.vocab_size, (2, 10)) # batch=2, seq_len=10
    
    # 训练/前向传播测试
    print("\n[测试前向传播与 Loss 计算]")
    logits, loss, _ = model(dummy_input_ids, labels=dummy_input_ids)
    print(f"Logits Shape: {logits.shape}")
    print(f"Sample Loss: {loss.item():.4f}")
    
    # 文本生成测试 (利用 KV Cache)
    print("\n[测试文本生成 (KV Cache 启用)]")
    start_tokens = torch.tensor([[1, 15, 20]], dtype=torch.long) # 假设 1 是 BOS
    generated_ids = model.generate(
        input_ids=start_tokens,
        max_new_tokens=20,
        temperature=0.8,
        top_k=40,
        top_p=0.9
    )
    print(f"生成的序列: {generated_ids.tolist()}")