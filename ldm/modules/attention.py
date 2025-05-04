from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

#   定义了一个跨注意力（CrossAttention）模块，用于计算查询（query）和上下文（context）之间的注意力权重，并将上下文的加权和作为输出。
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 512
        context_dim = default(context_dim, query_dim)  #    context_dim存在则是context_dim。不存在则是query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)        #   根据查询 x，通过 self.to_q 对其进行投影
        context = default(context, x)       #   如果上下文 context 不存在，则将其设置为查询 x
        k = self.to_k(context)          #   根据上下文 context，通过 self.to_k 和 self.to_v 对其进行投影，得到键 k 和值 v。
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))     #   将查询 q、键 k 和值 v 重塑为多头注意力的形状

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale            #   计算查询和键之间的相似度得分 sim，并进行缩放操作

        if exists(mask):                #   如果存在掩码 mask，将其重塑为与相似度矩阵 sim 相匹配的形状，并将无效位置的相似度值设置为负无穷大。
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)      #   对相似度矩阵 sim 进行 softmax 操作，得到注意力权重 attn

        out = einsum('b i j, b j d -> b i d', attn, v)      #   将注意力权重 attn 与值 v 进行加权求和，得到输出 out
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)     #   将输出 out 进行重塑，并通过 self.to_out 进行投影和 Dropout 操作，得到最终的输出
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention      #   是一个跨注意力（CrossAttention）模块，它实际上是一个自注意力（Self-Attention）模块，用于处理输入 x
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)   #   是一个前馈神经网络模块，用于在注意力之后对输入 x 进行非线性变换
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,     #   另一个跨注意力模块，它根据上下文 context 对输入 x 进行注意力操作。如果 context 为 None，则它仍然是一个自注意力模块。
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)      #   self.norm1、self.norm2 和 self.norm3 是规范化层（Normalization Layer），用于对输入进行归一化操作。
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint        #   self.checkpoint 是一个布尔值，用于指示是否使用模型的参数进行检查点操作

    def forward(self, x, context=None): #   在输入 x 和上下文 context 的基础上，调用了 checkpoint 函数，传递了 _forward 方法和相应的参数。这样可以根据 self.checkpoint 的值来决定是否使用模型参数进行检查点操作。
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):    #   实际进行前向传播的方法
        x = self.attn1(self.norm1(x)) + x       #   对输入 x 进行规范化，并通过 self.attn1 进行注意力操作，将注意力结果与输入 x 相加。
        x = self.attn2(self.norm2(x), context=context) + x      #   再次对结果进行规范化，并通过 self.attn2 对其进行注意力操作，将注意力结果与之前的结果相加。
        x = self.ff(self.norm3(x)) + x          #   再次对结果进行规范化，并通过 self.ff 进行前馈神经网络操作，将结果与之前的结果相加
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
    #   接受输入 x 和上下文 context
    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape        #   获取输入 x 的形状信息，并将其保存在变量 b、c、h 和 w 中
        x_in = x                    #   将 x 保存在变量 x_in 中，这将在最后与处理后的结果相加。
        x = self.norm(x)            #   对输入 x 进行归一化（Normalization）操作
        x = self.proj_in(x)         #   对输入 x 进行投影（Projection）操作
        x = rearrange(x, 'b c h w -> b (h w) c')    #   将输入 x 的形状从 'b c h w' 转换为 'b (h w) c'，以便在后续的 Transformer 块中进行处理
        for block in self.transformer_blocks:   #   通过循环遍历 self.transformer_blocks 中的每一个块，并将输入 x 和上下文 context 作为参数传递给每个块的 forward 方法进行处理。
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in