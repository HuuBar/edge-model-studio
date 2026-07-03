import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, TypeVar, List, Optional, Union, Tuple, cast

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

T = TypeVar("T")


@dataclass
class Config:
    """静态草稿树的基础超参数定义"""
    matrix_top_k: int = 8  # 每个 Token 对应的候选检索列数 (K)
    tree_depth: int = 4    # 静态草稿树的最大深度


@dataclass
class Outputs:
    output_ids: torch.Tensor
    accepted_sequences: List[List[int]]
    total_steps: int
    mean_accepted_tokens: float
    prompt_tokens_per_sec: float
    generation_tokens_per_sec: float


class Tree:
    def __init__(self, data: int, children: Optional[List['Tree']] = None):
        self.data = data
        self.children = children if children is not None else []

    def __repr__(self) -> str:
        return f"Tree(data={self.data}, children_count={len(self.children)})"


def map_breadthfirst(tree: Tree, fn: Callable[[Tree], T]) -> List[T]:
    """
    高性能、非递归的广度优先遍历（BFS）实现。
    """
    results: List[T] = []
    queue = [tree]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        results.append(fn(node))
        queue.extend(node.children)
    return results


class TokenRecycling:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, device: Optional[Union[str, torch.device]] = None, **kwargs):
        """
        初始化 TokenRecycling 模块。去除了硬编码的 FP16，自适应兼容 bfloat16 或量化权重。
        """
        # 默认不强制指定半精度，允许通过 kwargs 传递 torch_dtype 
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        return cls(model, tokenizer, device)

    def __init__(self, model, tokenizer: PreTrainedTokenizer, device: Optional[Union[str, torch.device]] = None):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dtype = model.dtype
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # 初始化分配零矩阵
        self.adjacency_matrix = torch.zeros(
            (len(self.tokenizer), Config.matrix_top_k),
            dtype=torch.long,
            device=self.device,
        )
        self.should_speculate = True
        self.use_cache = True

        # 调优控制日志开关
        self.show_tokens = True  
        self.debug_draft_tree = False  

        # 静态草稿树骨架解析
        self.tree_template = self.static_tree()

        # 计算并缓存相对位置 ID 与注意力掩码矩阵 (移除 root 节点对应的第一行/列)
        self.relative_position_ids = torch.tensor(
            self.get_relative_position_ids(self.tree_template),
            dtype=torch.long,
            device=self.device,
        )[1:]
        
        self.tree_attention_mask = self.get_tree_attention_mask(
            self.tree_template, device=self.device
        ).bool()[1:, 1:]

    def print_full_adjacency_matrix(self):
        """
        全量邻接矩阵导出（仅用于专项 Debug）
        """
        mat = self.adjacency_matrix.detach().cpu().numpy()
        vocab_size, top_k = mat.shape
        print(f"\n[DEBUG] Full Adjacency Matrix (Vocab={vocab_size}, TopK={top_k})")
        for tid in range(vocab_size):
            if np.any(mat[tid] > 0):  # 过滤全零行，降低无意义的 IO 打印延迟
                row = ",".join(str(int(x)) for x in mat[tid])
                print(f"  {tid}: {row}")

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 150,
        hot_start: bool = False,
        silent: bool = False,
        stop_on_eos: bool = True,
        adjacency_matrix: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> Outputs:
        """
        Token Recycling 核心推理框架
        """
        if isinstance(prompt, transformers.tokenization_utils_base.BatchEncoding):
            input_ids = prompt.input_ids.to(device=self.device)
        elif isinstance(prompt, torch.Tensor):
            input_ids = prompt.to(device=self.device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        prompt_length = input_ids.shape[-1]
        guess_length = 0  
        total_guesses = 0
        past_key_values: Optional[DynamicCache] = DynamicCache() if self.use_cache else None
        
        generation_start_time = None
        cached_tree_layers = None
        accepted_seqs: List[List[int]] = []

        prompt_start_time = time.perf_counter()  # 升级为高精度计时器
        steps = 0

        if self.should_speculate:
            if not hot_start or adjacency_matrix is None:
                self.adjacency_matrix.fill_(0)
            else:
                self.adjacency_matrix.copy_(adjacency_matrix.to(self.device))

        # 主干解码循环
        while input_ids.shape[-1] - prompt_length - guess_length < max_new_tokens:
            steps += 1
            position_ids = self.get_position_ids(input_ids, guess_length)
            attention_mask = self.get_attention_mask(input_ids, guess_length)
            use_full_input_ids = not self.use_cache or input_ids.shape[-1] == prompt_length

            # 区分 Prefill 状态与增量采样状态
            inputs_slice = input_ids if use_full_input_ids else input_ids[..., -(guess_length + 1):]
            pos_slice = position_ids if (use_full_input_ids or position_ids is None) else position_ids[..., -(guess_length + 1):]
            att_slice = attention_mask if (use_full_input_ids or attention_mask is None) else attention_mask[..., -(guess_length + 1):, :]

            logits = self.model(
                inputs_slice,
                position_ids=pos_slice,
                attention_mask=att_slice,
                past_key_values=past_key_values,
                use_cache=self.use_cache,
            ).logits

            if guess_length > 0:
                total_guesses += 1
            if generation_start_time is None:
                generation_start_time = time.perf_counter()

            next_token_index = -1 - guess_length

            # 在设备侧直接执行 Top-K 索引更新，规避 H2D 传输延迟
            if not hot_start:
                self.adjacency_matrix[inputs_slice] = logits.topk(Config.matrix_top_k).indices.to(dtype=torch.long)
                hot_start = True
            else:
                update_slice = next_token_index if guess_length == 0 else slice(next_token_index, None)
                self.adjacency_matrix[input_ids[:, update_slice]] = (
                    logits[:, update_slice, :].topk(Config.matrix_top_k).indices.to(dtype=torch.long)
                )

            # 贪婪采样当前的基准 Token
            next_token = logits[:, next_token_index, :].argmax(dim=-1)

            if not silent:
                print(
                    next_token.item() if self.show_tokens else self.tokenizer.decode(next_token),
                    end=" " if self.show_tokens else "",
                )
                sys.stdout.flush()

            if self.should_speculate:
                input_length = input_ids.shape[-1] - guess_length
                guesses = input_ids[..., input_length - 1:]
                input_ids = torch.cat([input_ids[..., :input_length], next_token.unsqueeze(0)], dim=-1)
                accepted_seqs.append([0])

                if guess_length > 0:
                    guess_logits = logits[..., input_length - 1:, :] if use_full_input_ids else logits
                    guess_max = guess_logits.argmax(dim=-1)
                    
                    # 验证并裁决最长有效草稿分支
                    longest_sequence = self.get_longest_sequence(self.tree_template, guesses, guess_max)

                    if len(longest_sequence) > 0:
                        verified_ids = guess_max[..., longest_sequence[1:]]

                        if not silent:
                            colors = ["95", "94", "92", "93", "91"]
                            for idx, _id in enumerate(verified_ids.squeeze(0)):
                                foreground = colors[idx % len(colors)]
                                ch = self.tokenizer.decode(_id)
                                underline = ";4" if ch.isspace() else ""
                                val = _id if self.show_tokens else ch
                                print(f"\033[{foreground}{underline}m{val}\033[0m", end=" " if self.show_tokens else "")
                            sys.stdout.flush()

                        input_ids = torch.cat([input_ids, verified_ids], dim=-1)
                        self.update_cache(past_key_values, guess_length, longest_sequence)
                        accepted_seqs[-1].extend(longest_sequence[1:].tolist())
                    elif past_key_values:
                        past_key_values.crop(input_length)

                # 生成下一阶段的草稿树 Token 序列
                tree_tokens, cached_tree_layers = self.merge_sequence(
                    self.adjacency_matrix,
                    self.tree_template,
                    input_ids[..., -1],
                    cached_tree_layers,
                )

                new_guesses = tree_tokens[1:]
                input_ids = torch.cat([input_ids, new_guesses.unsqueeze(0)], dim=-1)
                guess_length = new_guesses.shape[-1]
            else:
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

            # 提前触发停止符（EOS）阻断检查
            if stop_on_eos and (input_ids[..., prompt_length:input_ids.shape[-1] - guess_length] == self.tokenizer.eos_token_id).any():
                break

        # ==================== 指标精密测算 ====================
        mean_accepted_tokens = 0.0
        if total_guesses > 0 and accepted_seqs:
            mean_accepted_tokens = float(sum(len(s) for s in accepted_seqs) / len(accepted_seqs))

        prompt_tokens_per_sec = 0.0
        generation_tokens_per_sec = 0.0

        if generation_start_time is not None:
            end_time = time.perf_counter()
            prompt_time = max(generation_start_time - prompt_start_time, 1e-6)
            prompt_tokens_per_sec = float(prompt_length / prompt_time)

            gen_time = max(end_time - generation_start_time, 1e-6)
            gen_tokens = max(input_ids.shape[-1] - prompt_length - guess_length, 0)
            generation_tokens_per_sec = float(gen_tokens / gen_time)

        if not silent:
            print("\n")
            if total_guesses > 0:
                print(f"[METRIC] Mean Accepted Tokens: {mean_accepted_tokens:.2f}")
            print(f"[METRIC] Prompt Throughput: {prompt_tokens_per_sec:.2f} tokens/sec")
            print(f"[METRIC] Generation Throughput: {generation_tokens_per_sec:.2f} tokens/sec")

        # 剥离未被最终确认的猜测残留位
        input_ids = input_ids[..., :input_ids.shape[-1] - guess_length]
        self.last_token = int(input_ids[0, -1])

        # 严格遵守裁剪上限约束
        eos_indices = torch.where(input_ids[..., prompt_length:] == self.tokenizer.eos_token_id)[-1]
        eos_index = eos_indices.min().item() + prompt_length if eos_indices.numel() > 0 else input_ids.shape[-1]

        trim_length = max(input_ids.shape[-1] - prompt_length - max_new_tokens, input_ids.shape[-1] - eos_index)
        if trim_length > 0:
            input_ids = input_ids[..., :-trim_length]
            while accepted_seqs and trim_length > 0:
                if len(accepted_seqs[-1]) <= trim_length:
                    trim_length -= len(accepted_seqs.pop())
                else:
                    accepted_seqs[-1] = accepted_seqs[-1][:-trim_length]
                    trim_length = 0

        return Outputs(
            output_ids=input_ids,
            accepted_sequences=accepted_seqs,
            total_steps=steps,
            mean_accepted_tokens=mean_accepted_tokens,
            prompt_tokens_per_sec=prompt_tokens_per_sec,
            generation_tokens_per_sec=generation_tokens_per_sec,
        )

    def save_step_outputs(self, step: int, save_dir: str, tree_tokens: torch.Tensor):
        """单步落盘快照"""
        mat = self.adjacency_matrix.detach().cpu().numpy()
        with open(os.path.join(save_dir, f"Step{step}_matrics.txt"), "w", encoding="utf-8") as f:
            for i in range(mat.shape[0]):
                if np.any(mat[i] > 0):
                    f.write(f"{i} " + " ".join(str(int(x)) for x in mat[i]) + "\n")

        with open(os.path.join(save_dir, f"Step{step}_tree.txt"), "w", encoding="utf-8") as f:
            f.write(",".join(str(i) for i in tree_tokens.view(-1).tolist()) + "\n")

    # ==================== 内部验证与位置计算细节 ====================

    @classmethod
    def get_longest_sequence(cls, tree: Tree, guess_ids: torch.Tensor, actual_ids: torch.Tensor) -> torch.Tensor:
        """
        验证猜测 Token 序列。优化掉了原版频繁在循环中进行大张量 H2D 传输的瓶颈。
        """
        device = guess_ids.device
        
        # 预先在设备侧展平或提取，仅把小规模的索引拓扑树映射在 CPU 驱动进行树状剪枝
        guess_ids_cpu = guess_ids.squeeze(0).cpu()
        actual_ids_cpu = actual_ids.squeeze(0).cpu()

        node_to_index = {}
        node_to_parent = {}

        def register_node(node):
            node_to_index[node] = len(node_to_index)
            for child in node.children:
                node_to_parent[child] = node

        map_breadthfirst(tree, register_node)

        node_to_depth = {}
        stack = list(reversed(tree.children))
        deepest = (None, -1)

        while stack:
            node = stack.pop()
            parent = node_to_parent[node]
            node_depth = node_to_depth.get(parent, 0) + 1
            node_to_depth[node] = node_depth

            g_idx = node_to_index[node]
            v_idx = node_to_index[parent]

            # 精确单点匹配校验
            if guess_ids_cpu[g_idx] != actual_ids_cpu[v_idx]:
                continue

            if node_depth > deepest[1]:
                deepest = (node, node_depth)

            stack.extend(reversed(node.children))

        longest: List[int] = []
        if deepest[0] is not None:
            curr = deepest[0]
            while curr is not None:
                longest.append(node_to_index[curr])
                curr = node_to_parent.get(curr, None)

        return torch.tensor(list(reversed(longest)), dtype=torch.long, device=device)

    @classmethod
    def update_cache(cls, cache: Optional[DynamicCache], guess_length: int, verified_indices: torch.Tensor):
        """
        KV Cache 切片更新。精确对接 transformers>=4.55 新底层接口。
        """
        if cache is None:
            return

        seq_len = cache.get_seq_length()
        input_length = seq_len - guess_length

        keep_indices = torch.cat([
            torch.arange(0, input_length, device=verified_indices.device, dtype=torch.long),
            verified_indices[1:] - 1 + input_length
        ], dim=0)

        # 现代版本抽象组件路径检测
        layers = getattr(cache, "layers", None)
        if layers is not None:
            for layer in layers:
                if layer is None:
                    continue
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is not None and torch.is_tensor(keys) and keys.numel() > 0:
                    layer.keys = keys.index_select(-2, keep_indices)
                if values is not None and torch.is_tensor(values) and values.numel() > 0:
                    layer.values = values.index_select(-2, keep_indices)
            return

        # 兼容旧版本旧结构
        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            for i, (k, v) in enumerate(zip(key_cache, value_cache)):
                if k is not None and torch.is_tensor(k) and k.numel() > 0:
                    key_cache[i] = k.index_select(-2, keep_indices)
                if v is not None and torch.is_tensor(v) and v.numel() > 0:
                    value_cache[i] = v.index_select(-2, keep_indices)
            return

        raise TypeError(f"Unsupported KV cache architecture parsed: {type(cache)}")

    def get_position_ids(self, input_ids: torch.Tensor, guess_length: int) -> Optional[torch.Tensor]:
        if guess_length == 0:
            return None
        input_length = input_ids.shape[-1] - guess_length
        return torch.cat([
            torch.arange(input_length, dtype=self.relative_position_ids.dtype, device=self.device),
            self.relative_position_ids + input_length - 1
        ]).unsqueeze(0)

    def get_attention_mask(self, input_ids: torch.Tensor, guess_length: int) -> Optional[torch.Tensor]:
        if guess_length == 0:
            return None
        input_length = input_ids.shape[-1]
        min_dtype = torch.finfo(self.dtype).min
        mask = torch.full((input_length, input_length), fill_value=min_dtype, dtype=self.dtype, device=self.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        if guess_length > 0:
            mask[:, :, -guess_length:, -guess_length:] = (~self.tree_attention_mask).to(dtype=self.dtype) * min_dtype
        return mask

    @classmethod
    def get_relative_position_ids(cls, tree: Tree) -> List[int]:
        depths = {tree: 0}
        def calc_depth(node):
            d = depths[node]
            for child in node.children:
                depths[child] = d + 1
            return d
        return map_breadthfirst(tree, calc_depth)

    @classmethod
    def get_tree_attention_mask(cls, tree: Tree, device=None) -> torch.Tensor:
        nodes = map_breadthfirst(tree, lambda x: x)
        n = len(nodes)
        node_to_index = {node: i for i, node in enumerate(nodes)}
        node_to_parent = {}
        for node in nodes:
            for child in node.children:
                node_to_parent[child] = node

        mask = torch.zeros((n, n), dtype=torch.long, device=device)
        for i, node in enumerate(nodes):
            mask[i, i] = 1
            current = node
            while current in node_to_parent:
                parent = node_to_parent[current]
                mask[i, node_to_index[parent]] = 1
                current = parent
        return mask

    @classmethod
    def merge_sequence(cls, M: torch.Tensor, tree: Tree, xt: torch.Tensor, cached_layer_indices: Optional[List[List[torch.Tensor]]]) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """依据邻接矩阵 M 与静态树架构合并草稿序列"""
        device = xt.device
        L = xt.to(dtype=torch.int, device=M.device)
        S: List[torch.Tensor] = []
        d = 0

        if cached_layer_indices is None:
            def build_indices(node, layers, node_to_depth):
                depth = node_to_depth[node] + 1
                for child in node.children:
                    node_to_depth[child] = depth
                if len(layers) <= depth:
                    layers.append([])
                layers[depth].append(torch.tensor([c.data for c in node.children], dtype=torch.long, device=M.device))

            layers_buf: List[List[torch.Tensor]] = [[tree.data]]
            map_breadthfirst(tree, lambda n: build_indices(n, layers_buf, {tree: 0}))
            layer_indices = layers_buf[1:-1]
        else:
            layer_indices = cached_layer_indices

        tree_depth = len(layer_indices)
        while d < tree_depth:
            xs = M[L]
            Lnext = torch.cat([x.index_select(0, idxs) for x, idxs in zip(xs, layer_indices[d])])
            S.append(L)
            L = Lnext
            d += 1

        S = torch.cat(S + [L])
        return S.to(device), layer_indices

    @staticmethod
    def static_tree() -> Tree:
        """
        Token Recycling 图 5 对应的标准 4 层静态决策树算子拓扑骨架
        """
        layers = [
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3], [0, 1, 2], [0, 1], [0], [], []],
            [[0, 1, 2, 3, 4, 5], [0, 1, 2], [0, 1], [0, 1], [0, 1], [0], [0, 1], [0, 1], [0, 1], [0, 1], [0], [0], [0], [0], [0], [], [], [], [], [], [], [], [], []],
            [[0, 1, 2], [0, 1], [0], [0], [0], [0], [0, 1], [0], [0], [0], [0], [0], [0], [0], [0], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        ]
        root = Tree(data=0)
        curr = [root]
        for layer in layers:
            new_curr = []
            for children, parent in zip(layer, curr):
                parent.children = [Tree(data=c) for c in children]
                new_curr.extend(parent.children)
            curr = new_curr
        return root