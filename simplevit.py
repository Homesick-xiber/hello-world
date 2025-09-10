import torch
import torch.nn as nn
from einops.layers.torch import EinopsToRearrange # 用于简化维度操作

# 1. Patch Embedding 和 Positional Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # 将图像分割成 patch 并进行投影
        x = self.proj(x)  # [B, C, H, W] -> [B, E, H_new, W_new]
        # 将空间维度展平并转置
        x = x.flatten(2)  # [B, E, L] (L = H_new * W_new)
        x = x.transpose(1, 2)  # [B, L, E]
        return x

# 2. Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 多头自注意力机制
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output  # 残差连接
        # 前馈网络
        x = x + self.mlp(self.norm2(x))  # 残差连接
        return x

# 3. Vision Transformer 完整的模型
class ViT(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (image_size // patch_size) ** 2
        
        # 添加一个可学习的类别标记（class token）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer 编码器堆栈
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 1. Patch Embedding
        x = self.patch_embed(x)
        B, N, E = x.shape  # B: batch size, N: num patches, E: embed dim

        # 2. Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # 4. Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 5. Get class token and normalize
        x = self.norm(x[:, 0]) # 提取类别标记的输出

        # 6. Classification Head
        x = self.head(x)
        
        return x

# 4. 运行示例
if __name__ == "__main__":
    # 创建一个随机输入图像（batch_size=1, channels=3, size=224x224）
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # 实例化 ViT 模型
    # patch_size=16, depth=6 (简化), num_heads=8
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=10, # 假设有10个类别
        embed_dim=256,
        depth=6,
        num_heads=8
    )
    
    # 将模型设置为评估模式
    model.eval()
    
    # 进行前向传播
    with torch.no_grad():
        output = model(dummy_image)
    
    print("模型输出张量形状:", output.shape)
    # 预期的输出形状应为 [1, num_classes]
    # 在这个例子中是 [1, 10]