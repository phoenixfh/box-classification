"""
Timm模型包装器

将timm模型包装为HuggingFace兼容的接口
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional


class TimmModelWrapper(nn.Module):
    """
    Timm模型包装器
    
    使timm模型兼容HuggingFace Trainer的接口
    """
    
    def __init__(self, timm_model: nn.Module, num_labels: int):
        """
        Args:
            timm_model: timm模型实例
            num_labels: 标签数量
        """
        super().__init__()
        self.model = timm_model
        self.num_labels = num_labels
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ImageClassifierOutput:
        """
        前向传播
        
        Args:
            pixel_values: 输入图像 [batch_size, channels, height, width]
            labels: 标签 [batch_size]
            
        Returns:
            ImageClassifierOutput包含loss和logits
        """
        # timm模型的前向传播
        logits = self.model(pixel_values)
        
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
    
    def from_pretrained(self, load_directory: str):
        """加载模型"""
        import os
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(model_path))
