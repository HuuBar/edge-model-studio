import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "mocked response"
    tokenizer.apply_chat_template.return_value = "formatted_chat_prompt"
    tokenizer.chat_template = None 
    return tokenizer

@pytest.fixture
def mock_local_model(mock_tokenizer):
    model_mock = MagicMock()
    model_mock.device = "cpu"
    model_mock.generate.return_value = [[1, 2, 3, 4, 5]] # 模拟生成的 token ids

    local_model = MagicMock()
    local_model.model = model_mock
    local_model.model_id = "dummy_model_id"
    local_model.model_revision = "main"
    local_model.tokenizer = mock_tokenizer
    return local_model

@pytest.fixture
def adapter_instance(mock_local_model):
    # 隔离外部网络请求与依赖
    with patch('adapter.ChatGenerationModelAdapter._parse_generation_config') as mock_parse:
        mock_parse.return_value = MagicMock()
        adapter = ChatGenerationModelAdapter(model=mock_local_model)
        # 手动注入规避 super().__init__ 可能带来的外部依赖
        adapter.model = mock_local_model.model
        adapter.model_id = mock_local_model.model_id
        adapter.tokenizer = mock_local_model.tokenizer
        adapter.generation_config = MagicMock()
        return adapter

def test_init_with_custom_kwargs(mock_local_model):
    """测试初始化时自定义配置的覆盖逻辑"""
    with patch('adapter.ChatGenerationModelAdapter._parse_generation_config') as mock_parse:
        mock_gen_config = MagicMock()
        mock_parse.return_value = mock_gen_config
        
        custom_chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        
        adapter = ChatGenerationModelAdapter(
            model=mock_local_model,
            generation_config={"max_new_tokens": 1024},
            chat_template=custom_chat_template
        )
        
        mock_gen_config.update.assert_called_once_with(max_new_tokens=1024)
        assert adapter.tokenizer.chat_template == custom_chat_template


def test_parse_generation_config(mock_local_model):
    """测试生成配置的解析与默认值填充"""
    with patch('modelscope.GenerationConfig') as MockGenConfig:
        mock_remote_config = MagicMock()
        mock_remote_config.to_dict.return_value = {"remote_key": "val"}
        MockGenConfig.from_pretrained.return_value = mock_remote_config
        
        # 临时绕过 init 中的调用，直接测试静态方法逻辑
        adapter = MagicMock(spec=ChatGenerationModelAdapter)
        adapter.model_id = "test_id"
        adapter.model_revision = "main"
        
        config = ChatGenerationModelAdapter._parse_generation_config(
            adapter, mock_local_model.tokenizer, mock_local_model.model
        )
        
        # 验证默认 token ID 填充
        assert config.eos_token_id == 2
        assert config.pad_token_id == 0
        assert config.max_new_tokens == 2048


def test_prepare_inputs_no_template(adapter_instance):
    """测试无 chat_template 时，直接返回 query"""
    adapter_instance.tokenizer.chat_template = None
    inputs = [{"data": ["hello world"]}]
    result = adapter_instance._prepare_inputs(inputs)
    assert result == ["hello world"]


def test_prepare_inputs_with_template(adapter_instance):
    """测试包含 system_prompt 和 chat_template 的组装逻辑"""
    adapter_instance.tokenizer.chat_template = "jinja2_template"
    inputs = [
        {"data": ["hello"], "system_prompt": "You are a helpful assistant."}
    ]
    
    result = adapter_instance._prepare_inputs(inputs)
    
    assert len(result) == 1
    assert result[0] == "formatted_chat_prompt"
    adapter_instance.tokenizer.apply_chat_template.assert_called_once()


def test_model_generate(adapter_instance):
    """测试模型推理调用链路及输出解码"""
    adapter_instance.tokenizer.return_value = {"input_ids": [[1, 2]]}
    
    prompts = ["prompt1"]
    infer_cfg = {"num_return_sequences": 1, "stop": ["<|im_end|>"]}
    
    responses, input_lengths = adapter_instance._model_generate(prompts, infer_cfg)
    
    # 验证生成的结构和 tokenizer decode 的调用
    assert len(responses) == 1
    assert responses[0] == ["mocked response"]
    assert input_lengths == [3] # mock_tokenizer.encode 默认返回长度为 3 的 list
    adapter_instance.model.generate.assert_called_once()


@patch('time.time', return_value=1600000000)
def test_predict(mock_time, adapter_instance):
    """测试端到端 Predict 的数据封装"""
    # Mock 内部依赖方法
    adapter_instance._prepare_inputs = MagicMock(return_value=["formatted"])
    adapter_instance._model_generate = MagicMock(return_value=([["model_output"]], [10]))
    
    inputs = [{"data": ["test query"]}]
    
    results = adapter_instance.predict(inputs)
    
    assert len(results) == 1
    res = results[0]
    
    # 验证 ChatCompletionResponse 格式规范
    assert res['model'] == "dummy_model_id"
    assert res['object'] == "chat.completion"
    assert len(res['choices']) == 1
    assert res['choices'][0]['message']['content'] == "model_output"
    assert res['usage']['prompt_tokens'] == 10
    assert res['usage']['total_tokens'] == 13 # 10 + 3(mock_tokenizer encode length)