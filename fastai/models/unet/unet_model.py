""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    # 变体配置字典
    VARIANT_CONFIGS = {
        'n': {'base_channels': 8,  'depth': 3},  # nano
        's': {'base_channels': 16, 'depth': 3},  # small (当前默认)
        'm': {'base_channels': 32, 'depth': 4},  # medium
        'l': {'base_channels': 64, 'depth': 4},  # large
        'x': {'base_channels': 96, 'depth': 5},  # xlarge
    }
    
    def __init__(self, n_channels, n_classes, bilinear=False, variant='s'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.variant = variant
        
        # 根据变体获取配置
        config = self.VARIANT_CONFIGS.get(variant, self.VARIANT_CONFIGS['s'])
        base_channels = config['base_channels']
        depth = config['depth']
        
        # 输入卷积
        self.inc = DoubleConv(n_channels, base_channels)
        
        # 下采样路径（编码器）
        self.down_layers = nn.ModuleList()
        in_ch = base_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** (i + 1))
            self.down_layers.append(Down(in_ch, out_ch))
            in_ch = out_ch
        
        # 上采样路径（解码器）
        factor = 2 if bilinear else 1
        self.up_layers = nn.ModuleList()
        
        # 反向遍历，从最深层开始
        channels_list = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        for i in range(depth):
            # 当前层的输入通道（来自下一层）
            in_ch = channels_list[depth - i]
            # 跳跃连接的通道（来自编码器对应层）
            skip_ch = channels_list[depth - i - 1]
            # 输出通道
            out_ch = skip_ch
            
            if bilinear:
                # bilinear: concat后的通道数 = in_ch + skip_ch
                # DoubleConv 输入 = in_ch + skip_ch, 输出 = out_ch
                self.up_layers.append(Up(in_ch + skip_ch, out_ch, bilinear))
            else:
                # transpose: concat后的通道数 = in_ch // 2 + skip_ch = in_ch (因为 skip_ch = in_ch // 2)
                self.up_layers.append(Up(in_ch, out_ch, bilinear))
        
        # 输出卷积
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # 输入卷积
        x1 = self.inc(x)
        
        # 编码器（下采样）- 保存跳跃连接
        skip_connections = [x1]
        x = x1
        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)
        
        # 解码器（上采样）- 使用跳跃连接
        skip_connections = skip_connections[:-1]  # 去掉最后一个（bottleneck）
        for i, up in enumerate(self.up_layers):
            skip = skip_connections[-(i+1)]
            x = up(x, skip)
        
        # 输出
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        
        for down in self.down_layers:
            down = torch.utils.checkpoint(down)
        
        for up in self.up_layers:
            up = torch.utils.checkpoint(up)
        
        self.outc = torch.utils.checkpoint(self.outc)