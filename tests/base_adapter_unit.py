import pytest
from unittest.mock import MagicMock, patch
import torch

class DummyLocalModel:
    def __init__(self):
        self.model = "dummy_model"
        self.model_id = "dummy_id"
        self.model_revision = "main"
        self.device = "cpu"
        self.tokenizer = "dummy_tokenizer"
        self.model_cfg = {"cfg_key": "cfg_value"}

class DummyCustomModel:
    def __init__(self):
        self.config = {"custom_cfg": "custom_value"}

class ConcreteModelAdapter(BaseModelAdapter):
    def predict(self, *args, **kwargs):
        return "predicted_result"


# 测试 BaseModelAdapter.__init__
def test_base_model_adapter_init():
    """
    测试 BaseModelAdapter 的初始化逻辑。
    涵盖：传入 None、传入 LocalModel、传入 CustomModel，以及传入不支持类型时的异常处理。
    """
    adapter_none = ConcreteModelAdapter(model=None, model_cfg={"test": 123})
    assert adapter_none.model_cfg == {"test": 123}

    local_model = DummyLocalModel()
    with patch('model_adapter.LocalModel', DummyLocalModel):
        adapter_local = ConcreteModelAdapter(model=local_model)
        assert adapter_local.model == "dummy_model"
        assert adapter_local.model_id == "dummy_id"
        assert adapter_local.device == "cpu"

    custom_model = DummyCustomModel()
    with patch('model_adapter.CustomModel', DummyCustomModel):
        adapter_custom = ConcreteModelAdapter(model=custom_model)
        assert adapter_custom.model_cfg == {"custom_cfg": "custom_value"}

    with pytest.raises(ValueError, match="Unsupported model type"):
        ConcreteModelAdapter(model="Unsupported String Model")


# 测试 BaseModelAdapter.predict
def test_base_model_adapter_predict():
    """
    测试 BaseModelAdapter 的 predict 方法。
    验证具体子类实现 predict 后能否正常调用，并且 @torch.no_grad() 装饰器生效。
    """
    adapter = ConcreteModelAdapter(model=None)
    
    with patch('torch.is_grad_enabled', return_value=True):
        # 实际调用时，torch.no_grad 会暂时将 is_grad_enabled 置为 False
        # 这里验证方法能被正常调用并返回预期结果
        result = adapter.predict()
        assert result == "predicted_result"


# 测试 initialize_model_adapter
@patch('model_adapter.get_model_adapter')
def test_initialize_model_adapter(mock_get_model_adapter):
    mock_task_cfg = MagicMock()

    mock_task_cfg.eval_type = 'local' 
    mock_task_cfg.api_url = None
    mock_task_cfg.generation_config = {"max_new_tokens": 10}
    mock_task_cfg.chat_template = "default"

    mock_benchmark = MagicMock()
    mock_benchmark.model_adapter = "vllm"
    mock_benchmark.output_types = ["vllm", "hf"]
    mock_benchmark.name = "dummy_benchmark"

    mock_base_model = MagicMock()

    # 准备工厂返回的 mock adapter 实例
    mock_adapter_instance = MagicMock()
    mock_adapter_cls = MagicMock(return_value=mock_adapter_instance)
    mock_get_model_adapter.return_value = mock_adapter_cls

    with patch('model_adapter.EvalType') as MockEvalType:
        MockEvalType.CUSTOM = 'custom'
        MockEvalType.SERVICE = 'service'
        
        result = initialize_model_adapter(mock_task_cfg, mock_benchmark, mock_base_model)

    mock_get_model_adapter.assert_called_once_with("vllm")
    
    mock_adapter_cls.assert_called_once_with(
        model=mock_base_model,
        generation_config={"max_new_tokens": 10},
        chat_template="default",
        task_cfg=mock_task_cfg
    )
    
    assert result == mock_adapter_instance