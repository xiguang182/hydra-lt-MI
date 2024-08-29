import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.checkpoint

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop = 0., proj_drop = 0):
        super().__init__()
        # dim has been modified to the dim of each head (embedding)
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.out_channel = dim * num_heads
        self.scale = dim **-0.5

        self.q = nn.Linear(self.out_channel, self.out_channel, bias = qkv_bias)
        self.kv = nn.Linear(self.out_channel, self.out_channel * 2, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.out_channel, self.out_channel)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        # batch, number of patches, Channel (actual patch embeding * heads * 3)
        # x provides the query, y provides the key and value
        B, N, C = x.shape
        assert C == self.out_channel, 'total dim of out_channel dismatch'
        # reshape into B N 3(QKV) heads(parallel attentions, H) each patch(P) 
        # note: H * P = C
        # after permute: 3 B H N P
        
        # B N H P then permute B H N P
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        # B N 2 H P then permute 2 B H N P
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)

        # B H N P
        k, v = kv.unbind(0)

        # B H N N 
        attn = (q @ k.transpose(-2,-1)) * self.scale

        if self.causal:        
            # Create a mask with shape [N, N]
            # Masking future positions (upper triangle of the matrix)
            causal_mask = torch.triu(torch.ones((N, N)), diagonal=0)

            # Convert mask to boolean (True where mask is 1, False where it's 0)
            causal_mask = causal_mask.bool()
            # Apply the causal mask: setting masked positions to -inf
            attn.masked_fill_(causal_mask, float('-inf'))

        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        # B H N P then transpose B N H P then reshape: B N C
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop = 0., proj_drop = 0, causal = False):
        super().__init__()
        # dim has been modified to the dim of each head (embedding)
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.out_channel = dim * num_heads
        self.scale = dim **-0.5
        self.causal = causal

        self.qkv = nn.Linear(self.out_channel, self.out_channel * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.out_channel, self.out_channel)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # batch, number of patches, Channel (actual patch embeding * heads * 3)
        B, N, C = x.shape
        assert C == self.out_channel, 'total dim of out_channel dismatch'
        # reshape into B N 3(QKV) heads(parallel attentions, H) each patch(P) 
        # note: H * P = C
        # after permute: 3 B H N P
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        # B H N P
        q, k, v = qkv.unbind(0)

        # B H N N 
        attn = (q @ k.transpose(-2,-1)) * self.scale
        
        if self.causal:        
            # Create a mask with shape [N, N]
            # Masking future positions (upper triangle of the matrix)
            causal_mask = torch.triu(torch.ones((N, N)), diagonal=1)

            # Convert mask to boolean (True where mask is 1, False where it's 0)
            causal_mask = causal_mask.bool()
            # Apply the causal mask: setting masked positions to -inf
            attn.masked_fill_(causal_mask, float('-inf'))

        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        # B H N P then transpose B N H P then reshape: B N C
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values = 1e-5, inplace = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))


    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias =(bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio = 4., qkv_bias = False,
            drop = 0., attn_drop = 0, init_values = 1e-5, drop_path = 0.,
            act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()
        self.out_channel = dim * num_heads
        self.norm1 = norm_layer(self.out_channel)
        self.attn = Attention(dim, num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop)
        self.ls1 = LayerScale(self.out_channel, init_values = init_values)
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(self.out_channel)
        self.mlp = Mlp(in_features=self.out_channel, hidden_features=int(self.out_channel * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(self.out_channel, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

if __name__ == "__main__":
    # _ = SimpleDenseNet()
    # a = torch.rand((5,769))
    # model = BiLSTMLayer() 
    # output, (hd, _) = model(a)
    # print(output.shape)
    # print(hd.shape)
    model = block()
    output = model(a,b,c)
    print(output.shape)
