import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.model_max_length = 1024
    # 模拟 tokenizer() 返回
    tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
    return tokenizer

@pytest.fixture
def mock_local_model(mock_tokenizer):
    model_mock = MagicMock()
    model_mock.config = MagicMock(n_positions=2048)
    model_mock.device = "cpu"
    model_mock.generation_config = MagicMock()
    # 模拟模型 forward 返回 logits: [batch, seq_len, vocab_size]
    model_mock.return_value = {'logits': torch.randn(1, 3, 50000)}
    
    local_model = MagicMock()
    local_model.model = model_mock
    local_model.model_id = "test-model"
    local_model.device = "cpu"
    local_model.tokenizer = mock_tokenizer
    return local_model


def test_multi_choice_max_length(mock_local_model):
    """测试 max_length 属性的多级回退解析逻辑"""
    adapter = MultiChoiceModelAdapter(mock_local_model)
    assert adapter.max_length == 2048  # 取自 model.config.n_positions

    delattr(mock_local_model.model.config, 'n_positions')
    assert adapter.max_length == 1024  # 取自 tokenizer.model_max_length

def test_multi_choice_get_logits(mock_local_model):
    """测试 _get_logits 提取最后一个 token 概率的逻辑"""
    adapter = MultiChoiceModelAdapter(mock_local_model)
    log_probs, info = adapter._get_logits(adapter.tokenizer, adapter.model, ["test prompt"])
    
    assert 'tokens' in info
    # 输出形状应为 [batch_size, vocab_size]
    assert log_probs.shape == (1, 50000)

@patch('time.time', return_value=1234567890)
def test_multi_choice_predict(mock_time, mock_local_model):
    """测试多选题完整预测流程和结果封装"""
    adapter = MultiChoiceModelAdapter(mock_local_model)
    
    # 模拟 _get_logits 输出及 tokenizer 对选项的编码
    mock_logits = torch.tensor([[0.1, 0.9]]) # 假设词表大小2
    adapter._get_logits = MagicMock(return_value=(mock_logits, {}))
    adapter.tokenizer.side_effect = lambda x, **kwargs: {'input_ids': [0] if x=='A' else [1]}

    inputs = [{'data': ['Question?'], 'multi_choices': ['A', 'B']}]
    results = adapter.predict(inputs)

    assert len(results) == 1
    # B 对应的 logit(0.9) > A(0.1)，应输出 B
    assert results[0]['choices'][0]['message']['content'] == 'B'

def test_continuation_encode_pair(mock_local_model):
    """测试 context 与 continuation 拼接及空格处理"""
    adapter = ContinuationLogitsModelAdapter(mock_local_model)
    # 模拟 tokenizer 固定返回递增 id 便于切片验证
    adapter.tokenizer.side_effect = lambda x, **kwargs: {'input_ids': list(range(len(x)))}
    
    ctx = "Hello "
    cont = "World"
    ctx_enc, cont_enc = adapter._encode_pair(ctx, cont)
    
    assert len(ctx_enc) == len("Hello")
    assert len(cont_enc) == len(" World") # 空格被移交给了 continuation

def test_continuation_loglikelihood(mock_local_model):
    """测试 loglikelihood 提取和计算分数"""
    adapter = ContinuationLogitsModelAdapter(mock_local_model)
    adapter._encode_pair = MagicMock(return_value=(torch.tensor([1,2]), torch.tensor([3])))
    
    # 模拟 logits: [batch, seq, vocab]
    adapter.model.return_value = (torch.zeros(1, 3, 10),) 
    
    inputs = [("context", "continuation")]
    scores = adapter.loglikelihood(inputs)
    
    assert len(scores) == 1
    assert isinstance(scores[0], float)

@patch('time.time', return_value=1234567890)
def test_continuation_predict(mock_time, mock_local_model):
    """测试延续生成预测的封装格式"""
    adapter = ContinuationLogitsModelAdapter(mock_local_model)
    adapter.loglikelihood = MagicMock(return_value=[-1.5, -2.0])
    
    inputs = [{'data': [("ctx1", "cont1"), ("ctx2", "cont2")]}]
    results = adapter.predict(inputs)
    
    assert len(results) == 1
    assert results[0]['choices'][0]['message']['content'] == [-1.5, -2.0]
    assert results[0]['object'] == 'chat.completion'