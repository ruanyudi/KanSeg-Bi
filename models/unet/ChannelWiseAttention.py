import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(ChannelWiseSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        # 用于生成 query, key, value
        self.query_conv = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1
        )
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # softmax 用于计算注意力
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Query, Key, Value
        query = self.query_conv(x).view(
            batch_size, C // self.reduction, -1
        )  # B x (C/reduction) x (H*W)
        key = (
            self.key_conv(x).view(batch_size, C // self.reduction, -1).permute(0, 2, 1)
        )  # B x (H*W) x (C/reduction)
        value = self.value_conv(x).view(batch_size, C, -1)  # B x C x (H*W)
        # Attention map
        attention = torch.bmm(query, key)  # B x (C/reduction) x (C/reduction)
        attention = self.softmax(attention)  # B x (C/reduction) x (C/reduction)
        # 加权 value
        out = torch.bmm(
            value.permute(0, 2, 1), attention.permute(0, 2, 1)
        )  # B x C x (H*W)
        out = out.permute(0, 2, 1)
        out = out.view(batch_size, C, H, W)

        # 叠加输入
        out = out + x

        return out


if __name__ == "__main__":
    model = ChannelWiseSelfAttention(in_channels=64, reduction=1)
    dummy_input = torch.randn(1, 64, 4, 4)
    pred = model(dummy_input)
    print(pred.shape)
