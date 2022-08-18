import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)

        return out

class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        #nn.LeakyReLU(inplace=True),
        nn.ReLU(inplace=True),
        # nn.Dropout(0.1),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim,num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(in_dim)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual

        residual = out
        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out

class CNNTransformer(nn.Module):
    def __init__(self,
                 patch_h=16,
                 patch_w=16,
                 num_heads=8,
                 num_layers=4,
                 emb_dim=1024,
                 mlp_dim=2048,
                 in_channels=1,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1
                 ):
        super(CNNTransformer, self).__init__()
        
        self.dconv_st = double_conv(in_channels, 64)
        
        self.dconv_up1 = double_conv(64, 128)
        self.dconv_up2 = double_conv(128, 64)
        self.dconv_up3 = double_conv(64, 32)
        self.dconv_up4 = double_conv(32, 1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.avgpool = nn.AvgPool2d(4)
        
        self.dconv_down1 = double_conv(1, 32)
        self.dconv_down2 = double_conv(64,128)
        self.dconv_down3 = double_conv(192, 64)

        self.embedding = nn.Conv1d(patch_h*patch_w, emb_dim, kernel_size=1, stride=1)
        self.pos_embedding = PositionEmbs(patch_h*patch_w, emb_dim, dropout_rate)

        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(emb_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(emb_dim)        
        self.classifier = nn.Linear(emb_dim, 64)
        self.last = nn.Conv2d(64, 1, 3, padding=1)
        
        self.mor1 = double_conv(1, 8)
        self.mor2 = double_conv(1, 8)
    def forward(self, x):

        x = self.dconv_st(x)
        x = self.maxpool(x)
        x = self.dconv_up1(x)

        x = self.upsample(x)
        x_cv1 = self.dconv_up2(x) 
     
        x = self.upsample(x_cv1)
        x_cv2 = self.dconv_up3(x)
        
        x = self.upsample(x_cv2)
        x_cv3 = self.dconv_up4(x)
        x = rearrange(x_cv3, 'b 1 (h m) (w n) -> b (m n) (h w)',m=16,n=16)
        x = rearrange(x, 'b c w -> b w c')
        x = self.embedding(x)
        x = rearrange(x, 'b w c -> b c w')

        x = self.pos_embedding(x)
        for layer in self.encoder_layers:
            out = layer(x)
        out = self.norm(out)

        out = self.classifier(out)
        out = rearrange(out, 'b (m n) (h w) -> b 1 (h m) (w n)', m=16, n=16,h=8,w=8)
        out = self.dconv_down1(out)
        out = torch.cat([out, x_cv2], dim=1)
        out = self.dconv_down2(out)
        out = self.maxpool(out)
        out = torch.cat([out, x_cv1], dim=1)
        out = self.dconv_down3(out)
        out = self.last(out)
        return out