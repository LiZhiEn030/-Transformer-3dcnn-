import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3),
                              stride=(1,stride,stride), padding=(1,1,1))
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=(3,3,3),
                              padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1,
                         stride=(1,stride,stride)),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)



class SpatioTemporalAttention(nn.Module):
    def __init__(self, in_dim=256, num_heads=8):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        spatial_out, _ = self.spatial_attn(x, x, x)
        B_T, HW, C = x.shape
        temporal_input = x.view(-1, 16, HW, C).permute(0,2,1,3).reshape(-1,16,C)
        temporal_out, _ = self.temporal_attn(temporal_input, temporal_input, temporal_input)
        temporal_out = temporal_out.view(B_T//16, HW, 16, C).permute(0,2,1,3).reshape(B_T, HW, C)
        return self.gate * spatial_out + (1 - self.gate) * temporal_out

class HybridModel(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),
            ResidualBlock3D(64, 128, stride=2),
            ResidualBlock3D(128, 256, stride=2)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.spatio_temp_attn = SpatioTemporalAttention()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_out = self.cnn(x)  # (B,256,16,14,14)
        B, C, T, H, W = cnn_out.shape
        spatial_features = cnn_out.permute(0,2,3,4,1).reshape(B*T, H*W, C)
        attn_out = self.spatio_temp_attn(spatial_features)
        attn_out = attn_out.contiguous().view(B, T*H*W, C)
        trans_out = self.transformer(attn_out)
        trans_out = trans_out.view(B, T, H, W, C).permute(0,4,1,2,3)
        return self.classifier(trans_out)

def test_model():
    model = HybridModel(num_classes=101)
    dummy_input = torch.randn(8, 3, 16, 224, 224)  # (B,C,T,H,W)

    # 验证分类头输入维度
    cnn_out = model.cnn(dummy_input)  # (8,256,16,14,14)
    trans_out = model.transformer(...)  # 假设正常输出
    classifier_input = trans_out.view(8, 16, 14, 14, 256).permute(0, 4, 1, 2, 3)
    print("分类头输入形状:", classifier_input.shape)  # 应输出 (8,256,16,14,14)

    output = model(dummy_input)
    print("最终输出形状:", output.shape)  # 应输出 (8,101)


if __name__ == "__main__":
    test_model()