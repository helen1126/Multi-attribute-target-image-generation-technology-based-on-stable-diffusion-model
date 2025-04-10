import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.mha import TextEncoderWithMHA
import argparse

# 双卷积模块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 空间-语义交叉注意力模块
class SpatialSemanticAttention(nn.Module):
    def __init__(self, in_channels, semantic_dim):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.W_q = nn.Linear(semantic_dim, in_channels)
        self.W_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_o = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, semantic_embedding):
        b, c, h, w = x.shape
        # 计算查询
        Q = self.W_q(semantic_embedding).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        # 计算键和值
        K = self.W_k(x)
        V = self.W_v(x)

        # 计算注意力分数
        scores = torch.sum(Q * K, dim=1, keepdim=True) / (self.semantic_dim ** 0.5)

        attention_weights = torch.softmax(scores, dim=-1)

        # 计算注意力输出
        attn_output = attention_weights * V
        attn_output = self.W_o(attn_output)

        # 融合原始特征图和注意力输出
        output = x + attn_output

        return output

# UNet模块
class UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], semantic_dim=768
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 下采样部分
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 上采样部分
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            self.ups.append(SpatialSemanticAttention(feature, semantic_dim))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, semantic_embedding):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 3]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
            x = self.ups[idx + 2](x, semantic_embedding)

        return self.final_conv(x)

'''临时测试函数
def test_u_net():
    parser = argparse.ArgumentParser(description='UNet with Attention')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=3, help='Number of output channels')
    args = parser.parse_args()

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化文本编码器
    text_encoder = TextEncoderWithMHA()
    text = ["dog"]  # 输入文本
    result = text_encoder.encode_text(text)
    semantic_embedding = result["embeddings"].to(device)

    model = UNet(in_channels=args.in_channels, out_channels=args.out_channels).to(device)
    # 加载图片
    image_path = 'image3.jpg'  # 输入图像地址
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    # 模型推理
    output = model(input_image, semantic_embedding)
    output = output.cpu().squeeze(0).detach().numpy()

    # 归一化到 [0, 1] 范围
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    # 调整维度以符合 imshow 要求 (H, W, C)
    if args.out_channels == 1:
        output = output.squeeze()
    else:
        output = output.transpose(1, 2, 0)

    # 显示和保存结果
    plt.imshow(output)
    plt.axis('off')
    plt.savefig('output_image.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    test_u_net()
'''