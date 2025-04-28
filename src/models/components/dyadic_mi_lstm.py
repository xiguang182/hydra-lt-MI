"""
A hybrid model with a transformer encoder for openface features and a lstm for RoBERTa features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.checkpoint
# from src.models.components.model_helper import 
import numpy as np
def positional_embedding(n, dim):
    pe = torch.zeros(n, dim)
    position = torch.arange(0, n).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe



class BiLSTMLayer(nn.Module):
    """A BiLSTM layer.
    In the case of the base line: 
    There are BiLSTM layers with 300 hidden units.
    return the hidden state of the last layer
    dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1
    """

    def __init__(
            self, 
            input_size: int = 769, 
            hidden_size: int = 300,
            num_layers: int = 1,
            dropout: float = 0.,
        ) -> None:
        """Initialize a `BiLSTMLayer` module.

        :param input_size: The number of input features.
        :param hidden_size: The number of hidden units.
        :param num_layers: The number of LSTM layers.
        :param dropout: The dropout rate.
        """
        self.hidden_size = hidden_size
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        # hiden state and cell state are not used
        # x, (_, _) = self.bilstm(x)
        # the base line uses the hidden layer as the output
        ret, (hid_state, _) = self.bilstm(x)
        # change the shape to B, Bi * num_layers, hidden_size
        hid_state = hid_state.permute(1,0,2)     
        # # manually batch first B, Sequence length, Bi * hidden_size
        # ret = ret.permute(1,0,2)
        # take the forward sequence of LSTM output
        ret = ret[:,:,0: self.hidden_size]
        return (hid_state, ret)
    

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
        if False:
            print(f"X shape: {x.shape}")
            print(f"Y shape: {y.shape}")
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
            causal_mask = torch.triu(torch.ones((N, N)), diagonal=1).to(x.device)

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
        # layerNorm device issue.
        # print(f"Input device: {x.device}")
        # print(f"LayerNorm weight device: {self.norm1.weight.device}")
        # print(f"LayerNorm bias device: {self.norm1.bias.device}")
        # self.norm1 = self.norm1.to(x.device)
        
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # is a mlp layer here necessary?
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ClassiferLayer(nn.Module):
    """A classifier layer.
    In the case of the base line: 
    There is a classifier layer with 300 input features and 2 output classes.
    """

    def __init__(
            self, 
            in_features: int = 300, 
            out_features: int = 2,
        ) -> None:
        """Initialize a `ClassiferLayer` module.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.softmax(self.linear(x))

class LSTMT(nn.Module):
    """ 
    LSTM on text as stream 1, 
    self attention on text as stream 2,
    cross attention from client to stream 2, 
    cross attention from counseller to stream 2,
    """
    def __init__(self, dim = 512, depth = 3, num_heads = 8, mlp_ratio = 4, 
                 qkv_bias = False, drop = 0, attn_drop = 0, init_values = 1e-5, 
                 drop_path = False, act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False):
        super().__init__()
        self.dim = dim

        text_dim = 769
        client_dim = 674
        counseller_dim = 674


        self.bilstm_RoBERTa = BiLSTMLayer(
            input_size=text_dim,
            hidden_size=dim,
            num_layers=1,
        )

        self.classifier = ClassiferLayer(dim * 2, 2)



    def forward(self, x, y, z):
        lstm_ret = self.bilstm_RoBERTa(x)[0]
        lstm_ret = lstm_ret.flatten(start_dim=1)
        return self.classifier(lstm_ret)

class LSTMT_cCl_cCo_Model(nn.Module):
    """ 
    LSTM on text as stream 1, 
    self attention on text as stream 2,
    cross attention from client to stream 2, 
    cross attention from counseller to stream 2,
    """
    def __init__(self, dim = 512, depth = 3, num_heads = 8, mlp_ratio = 4, 
                 qkv_bias = False, drop = 0, attn_drop = 0, init_values = 1e-5, 
                 drop_path = False, act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        text_dim = 769
        client_dim = 674
        counseller_dim = 674
        self.linear_text = nn.Linear(text_dim, dim)
        self.linear_client = nn.Linear(client_dim, dim)
        self.linear_counseller = nn.Linear(counseller_dim, dim)


        self.bilstm_RoBERTa = BiLSTMLayer(
            input_size=text_dim,
            hidden_size=dim,
            num_layers=1,
        )

        self.self_blocks = nn.ModuleList([
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

        self.cl_blocks = nn.ModuleList([Cross_block(   # cross attention from client to text    
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
        ) for i in range(depth)])

        self.co_blocks = nn.ModuleList([Cross_block(   # cross attention from counseller to text    
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
        ) for i in range(depth)])

        self.classifier = ClassiferLayer(dim * 3, 2)



    def forward(self, x, y, z):
        lstm_ret = self.bilstm_RoBERTa(x)[0]
        lstm_ret = lstm_ret.flatten(start_dim=1)
        
        # positional embedding, may cause device error
        _, N, _ = x.shape
        x = self.linear_text(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1)
        x = x + positional_embedding(N + 1, self.dim).to(x.device)
        _, M, _ = y.shape
        y = self.linear_client(y) + positional_embedding(M, self.dim).to(y.device)
        _, O, _ = z.shape
        z = self.linear_counseller(z) + positional_embedding(O, self.dim).to(z.device)
        
        for block in self.self_blocks:
            x = block(x)
        for block in self.cl_blocks:
            x = block(x, y)
        for block in self.co_blocks:
            x = block(x, z)
        # x = self.sT(x)
        x = torch.cat((lstm_ret, x[:,0]), dim=1)

        return self.classifier(x)
    

class LSTMT_lstmT_cCl_cCo_Model(nn.Module):
    """ 
    LSTM on text, take hidden state as stream 1, take output as stream 2,
    cross attention from client to stream 2, 
    cross attention from counseller to stream 2,
    """
    def __init__(self, dim = 512, depth = 1, num_heads = 8, mlp_ratio = 4, 
                 qkv_bias = False, drop = 0, attn_drop = 0, init_values = 1e-5, 
                 drop_path = False, act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        text_dim = 769
        client_dim = 674
        counseller_dim = 674
        self.linear_client = nn.Linear(client_dim, dim)
        self.linear_counseller = nn.Linear(counseller_dim, dim)


        self.bilstm_RoBERTa = BiLSTMLayer(
            input_size=text_dim,
            hidden_size=dim,
            num_layers=1,
        )


        self.cl_blocks = nn.ModuleList([Cross_block(   # cross attention from client to text    
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
        ) for i in range(depth)])

        self.co_blocks = nn.ModuleList([Cross_block(   # cross attention from counseller to text    
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
        ) for i in range(depth)])

        self.classifier = ClassiferLayer(dim * 3, 2)



    def forward(self, x, y, z):
        lstm_hid, lstm_ret = self.bilstm_RoBERTa(x)
        lstm_hid = lstm_hid.flatten(start_dim=1)
        
        # positional embedding, may cause device error
        _, N, _ = lstm_ret.shape
        lstm_ret = torch.cat((self.cls_token.expand(lstm_ret.shape[0], -1, -1), lstm_ret), dim = 1)
        lstm_ret = lstm_ret + positional_embedding(N + 1, self.dim).to(lstm_ret.device)
        _, M, _ = y.shape
        y = self.linear_client(y) + positional_embedding(M, self.dim).to(y.device)
        _, O, _ = z.shape
        z = self.linear_counseller(z) + positional_embedding(O, self.dim).to(z.device)
        
        
        for block in self.cl_blocks:
            lstm_ret = block(lstm_ret, y)
        for block in self.co_blocks:
            lstm_ret = block(lstm_ret, z)
        # x = self.sT(x)
        x = torch.cat((lstm_hid, lstm_ret[:,0]), dim=1)

        return self.classifier(x)


class sT(nn.Module):
    """
    self attention on text
    """
    def __init__(self, dim = 512, depth = 3, num_heads = 8, mlp_ratio = 4, 
                 qkv_bias = False, drop = 0, attn_drop = 0, init_values = 1e-5, 
                 drop_path = False, act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        text_dim = 769
        self.linear_text = nn.Linear(text_dim, dim)

        self.self_blocks = nn.ModuleList([
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

        self.classifier = ClassiferLayer(dim, 2)

    def forward(self, x, _y, _z):
        # positional embedding, may cause device error
        _, N, _ = x.shape
        x = self.linear_text(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1)
        x = x + positional_embedding(N + 1, self.dim).to(x.device)

        for block in self.self_blocks:
            x = block(x)
        # x = self.sT(x)
        return self.classifier(x[:,0])
    
class sT_cCl(nn.Module):
    """
    self attention on text
    cross attention from client to text
    """
    def __init__(self, dim = 512, depth = 6, num_heads = 8, mlp_ratio = 4, 
                 qkv_bias = False, drop = 0, attn_drop = 0, init_values = 1e-5, 
                 drop_path = False, act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        text_dim = 769
        client_dim = 674
        self.linear_text = nn.Linear(text_dim, dim)
        self.linear_client = nn.Linear(client_dim, dim)

        self.self_blocks = nn.ModuleList([
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

        self.cl_blocks = nn.ModuleList([Cross_block(   # cross attention from client to text    
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
        ) for i in range(depth)])

        self.classifier = ClassiferLayer(dim, 2)

    def forward(self, x, y):
        # positional embedding, may cause device error
        _, N, _ = x.shape
        x = self.linear_text(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1)
        x = x + positional_embedding(N + 1, self.dim).to(x.device)

        _, M, _ = y.shape
        y = self.linear_client(y) + positional_embedding(M, self.dim).to(y.device)

        for block in self.self_blocks:
            x = block(x)
        for block in self.cl_blocks:
            x = block(x, y)

        return self.classifier(x[:,0])

if __name__ == "__main__":
    a = torch.rand((5,5,769))
    b = torch.rand((5,6,674))
    # model = Cross_block(dim=128, causal=True)
    # output = model(a, b)
    model = LSTMT(dim=128)
    output = model(a, b, b)
    print(output.shape)
