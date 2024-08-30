import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.checkpoint

class CrossAttention(nn.Module):
    """ Cross Attention layer.
    Cross Attention from Feature Y to X. Yeilds a new feature X' with the attention combination of Y.
    X -> Q, Y -> K, V
    out = linear(mask(layerscale(Q(K^T)))V)
    """
    def __init__(
            self, 
            dim, num_heads = 8, 
            qkv_bias = False, 
            attn_drop = 0., 
            proj_drop = 0, 
            causal = False
        ) -> None:
        """Initialize a `CrossAttention` module.
        :param dim: The number of each patch.
        :param num_heads: The number of attention heads.
        :param qkv_bias: Whether to use bias in the qkv projection.
        :param attn_drop: The dropout rate for the attention weights.
        :param proj_drop: The dropout rate for the output tensor.
        :param casual: Whether to use a causal mask.
        """
        super().__init__()
        # dim has been modified to the dim of each head (embedding)
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.out_channel = dim * num_heads
        self.scale = dim **-0.5
        self.causal = causal

        self.q = nn.Linear(self.dim, self.out_channel, bias = qkv_bias)
        self.kv = nn.Linear(self.dim, self.out_channel * 2, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.out_channel, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.
        :param x: The target tensor.
        :param y: The referencing tensor.
        :return: The target tensor represented by referencing tensor.
        """
        # batch, number of patches, dim of each patch
        # x provides the query, y provides the key and value
        B, N, P = x.shape
        _, M, _ = y.shape
        assert P == self.dim, 'dimension dismatch'
        
        # B N H P then permute B H N P
        q = self.q(x).reshape(B, N, self.num_heads, P).permute(0,2,1,3)
        # B N 2 H P then permute 2 B H M P
        kv = self.kv(y).reshape(B, M, 2, self.num_heads, P).permute(2,0,3,1,4)

        # B H M P
        k, v = kv.unbind(0)

        # B H N P * B H P M -> B H N M 
        attn = (q @ k.transpose(-2,-1)) * self.scale

        if self.causal:        
            # Create a mask with shape [N, M]
            # Masking future positions (upper triangle of the matrix)
            causal_mask = torch.triu(torch.ones((N, M)), diagonal=1)

            # Convert mask to boolean (True where mask is 1, False where it's 0)
            causal_mask = causal_mask.bool()
            # Apply the causal mask: setting masked positions to -inf
            attn.masked_fill_(causal_mask, float('-inf'))

        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        #  B H N M * B H M P -> B H N P 
        # then transpose B N H P then reshape: B N H*P, the output of the attention
        x = (attn @ v).transpose(1,2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_block(nn.Module):
    """Cross Attention Block
    A complete cross attention block:
    include layer scale, drop path, norm layer, mlp layer and residual connection between the cross attention layer
    Drop path is not implemented as the model is not deep enough to require it.
    """
    def __init__(
            self, 
            dim, num_heads = 8, mlp_ratio = 4., qkv_bias = False,
            drop = 0., attn_drop = 0., init_values = 1e-5, drop_path = 0.,
            act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False
        ) -> None:
        """Initialize a `cross_block` module.
        :param dim: The number of each patch.
        :param num_heads: The number of attention heads.
        :param mlp_ratio: The ratio of the hidden layer/input layer in the mlp layer.
        :param qkv_bias: Whether to use bias in the qkv projection.
        :param drop: The dropout rate for the mlp layer.
        :param attn_drop: The dropout rate for the attention weights.
        :param init_values: The initial value for the layer scale.
        :param drop_path: The dropout rate for the drop path.
        :param act_layer: The activation function for the mlp layer.
        :param norm_layer: The normalization layer.
        :param causal: Whether to use a causal mask.
        """
        super().__init__()
        self.dim = dim
        self.norm_x = norm_layer(self.dim)
        self.norm_y = norm_layer(self.dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop, causal=causal)
        self.ls1 = LayerScale(self.dim, init_values = init_values)
        # omited the drop path
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = nn.Identity()

        self.norm_out = norm_layer(self.dim)
        self.mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(self.dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x, y):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm_x(x), self.norm_y(y))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm_out(x))))
        return x

class Attention(nn.Module):
    """ Self Attention layer.
    takes B, N, C and returns B, N, C, where C is the dim of each patch
    C first projected to num_heads copies of Q, K, V then reshaped to B, N, 3, H, P, 
    where H is the number of heads, P is the dim of each head
    """
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop = 0., proj_drop = 0, causal = False):
        super().__init__()
        # dim has been modified to the dim of each head (embedding)
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.out_channel = dim * num_heads
        self.scale = dim **-0.5
        self.causal = causal

        self.multihead_qkv = nn.Linear(self.dim, self.out_channel * 3, bias = qkv_bias)
        self.qkv = nn.Linear(self.out_channel, self.out_channel * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.out_channel, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # batch, number of patches, dim of each patch
        B, N, C = x.shape
        assert C == self.dim, 'dim of out_channel dismatch'
        # reshape into B N 3(QKV) heads(parallel attentions, H) each patch(P) 
        # note: H * P = C
        # after permute: 3 B H N P
        qkv = self.multihead_qkv(x).reshape(B, N, 3, self.num_heads, C).permute(2,0,3,1,4)
        # B H N P
        q, k, v = qkv.unbind(0)

        # B H N N 
        attn = (q @ k.transpose(-2,-1)) * self.scale
        
        if self.causal:        
            # Create a mask with shape [N, N]
            # Masking future positions (upper triangle of the matrix)
            # to device?
            causal_mask = torch.triu(torch.ones((N, N)), diagonal=1)

            # Convert mask to boolean (True where mask is 1, False where it's 0)
            causal_mask = causal_mask.bool()
            # Apply the causal mask: setting masked positions to -inf
            attn.masked_fill_(causal_mask, float('-inf'))

        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        # B H N P then transpose B N H P then reshape: B N C*H
        x = (attn @ v).transpose(1,2).reshape(B, N, -1)
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

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(
            self, dim, num_heads = 8, mlp_ratio = 4., qkv_bias = False,
            drop = 0., attn_drop = 0., init_values = 1e-5, drop_path = 0.,
            act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(self.dim)
        self.attn = Attention(dim, num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop, causal=causal)
        self.ls1 = LayerScale(self.dim, init_values = init_values)
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(self.dim)
        self.mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(self.dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # is a mlp layer here necessary?
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class alt_SSTransformer(nn.Module):
    def __init__(
        self, window_frame, patch_frame, num_heads = 8, embed_dim=16, depth = 6, drop_out = 0, qkv_bias = True, init_values = 1e-5, modality_stacks = 1):
        super().__init__()

        self.attn_out = 7
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU
        self.act = nn.GELU()
        embed_dim = embed_dim // 16
        # heads * embed_dim = total dim for multi-head attation i.e. C in embed output (B, L, C)
        self.out_channel = embed_dim * num_heads
        self.patch_embed = nn.Linear(self.out_channel * 2, self.out_channel)
        self.embed_mlp =  Mlp(in_features=self.out_channel * 2, hidden_features=int(self.out_channel * 2 * 2), out_features = self.out_channel)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.out_channel))
        # actual number of embeddings + class 1 token, in this case it will be used for regression output.
        num_embed = modality_stacks * window_frame // patch_frame + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_embed, self.out_channel) * .02)
        self.pos_drop = nn.Dropout(p = drop_out)
        # self.drop_out = drop_out
        self.drop_out = drop_out
        dpr = [x.item() for x in torch.linspace(0, drop_out, depth)]  # stochastic depth decay rule

        self.blocks =  nn.Sequential(*[
            block(
                dim=embed_dim, num_heads = num_heads, mlp_ratio=4, qkv_bias=qkv_bias, init_values=init_values,
                drop= self.drop_out, attn_drop= drop_out, drop_path=dpr[i], norm_layer=norm_layer, act_layer = act_layer)
            for i in range(depth)])

        self.norm = norm_layer(self.out_channel)
        # no norm for class token

        # attn Head, output from each attn stream
        self.head = nn.Linear(self.out_channel, self.attn_out)

    def init_weight(self):
        return None

    # pos embeding AND prepend class token
    def _pos_embed(self, x):
        # -1 means not changing size
        # x is from embed (B, L, C)
        # cls token is (1, 1, C)
        
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_feature(self, x):
        # x = self.act(self.patch_embed(x))
        x = self.embed_mlp(x)
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        x = x[:,0]
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)
        x = self.forward_head(x)
        return x
    
class sT_cCl_cCo_Model(nn.Module):
    """ 
    self attention on text, 
    cross attention from client to text, 
    cross attention from counseller to text
    """
    def __init__(self, dim, depth = 1, num_heads = 8, mlp_ratio = 4, 
                 qkv_bias = False, drop = 0, attn_drop = 0, init_values = 1e-5, 
                 drop_path = False, act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads


        self.sT = Block(
            dim = dim, 
            num_heads = num_heads, 
            mlp_ratio = mlp_ratio, 
            qkv_bias = qkv_bias, 
            drop = drop, 
            attn_drop = attn_drop, 
            init_values = init_values, 
            drop_path = drop_path, 
            act_layer = act_layer, 
            norm_layer = norm_layer, 
            causal = causal
        )
        self.blocks = nn.ModuleList([
            Block(
                dim = dim, 
                num_heads = num_heads, 
                mlp_ratio = mlp_ratio, 
                qkv_bias = qkv_bias, 
                drop = drop, 
                attn_drop = attn_drop, 
                init_values = init_values, 
                drop_path = drop_path, 
                act_layer = act_layer, 
                norm_layer = norm_layer, 
                causal = causal
            ) for i in range(depth)
        ])
    def forward(self, x, y, z):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    a = torch.rand((5,5,128))
    b = torch.rand((5,6,128))
    model = Cross_block(dim=128, causal=True)
    output = model(a, b)
    print(output.shape)