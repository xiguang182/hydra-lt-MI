"""
A hybrid model with a transformer encoder for openface features and a lstm for RoBERTa features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.checkpoint
# from src.models.components.model_helper import 
import numpy as np
import os
def positional_embedding(n, dim):
    pe = torch.zeros(n, dim)
    position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe



class BiLSTMLayer(nn.Module):
    """A BiLSTM layer.
    In the case of the base line: 
    There are BiLSTM layers with 300 hidden units.
    return the hidden state of the last layer
    dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1
    return (hidden state, output)
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
    
class LinearLayer(nn.Module):
    """A linear layer.
    In the case of the base line. There are 2 linear layers:
    one with 600 input features and 200 output features 
    the other with 800 input features and 300 output features.
    """

    def __init__(
            self, 
            in_features: int = 784, 
            out_features: int = 10,
            activation = nn.ReLU,
        ) -> None:
        """Initialize a `LinearLayer` module.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param activation: The activation function.
        """
        super().__init__()
        self.activation = activation()
        self.linear = nn.Linear(in_features, out_features)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.activation(self.linear(x))

class ClassiferLayer(nn.Module):
    """A classifier layer.
    In the case of the base line: 
    There is a classifier layer with 300 input features and 2 output classes.
    """

    def __init__(
            self, 
            in_features: int = 300, 
            out_features: int = 2,
            ifsoftmax = True,
            act_layer = nn.ReLU,
            drop = 0.1,
        ) -> None:
        """Initialize a `ClassiferLayer` module.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        """
        super().__init__()
        self.mlp = Mlp(
            in_features=in_features,
            hidden_features=int(in_features / 2),
            out_features=out_features,
            act_layer=act_layer,
            drop=drop
        )
        self.ifsoftmax = ifsoftmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.mlp(x)
        
        if self.ifsoftmax:
            return self.softmax(x)
        else:
            return x


# used for pretrained weights
class LSTMT(nn.Module):
    """ 
    LSTM on text alone
    """
    def __init__(self, dim = 512):
        super().__init__()
        self.dim = dim

        text_dim = 769


        self.bilstm_RoBERTa = BiLSTMLayer(
            input_size=text_dim,
            hidden_size=dim,
            num_layers=1,
        )

        self.classifier = ClassiferLayer(dim * 2, 2,ifsoftmax=False)



    def forward(self, x, y, z):
        lstm_ret = self.bilstm_RoBERTa(x)[0]
        lstm_ret = lstm_ret.flatten(start_dim=1)
        return self.classifier(lstm_ret)

# used for pretrained weights
class LSTMT_LSTMCl(nn.Module):
    """The base line model. 
    1. 6 * (768 RoBERTa feature + 1 speaker ID)
    2. 6 * 674 openface feature for client 
    """

    def __init__(self, dim = 512) -> None:
        """Initialize a `BaseLineModel` module.

        :param num_classes: The number of classes.
        :param lstm_input_size: The number of input features for the BiLSTM layer.
        :param lstm_hidden_size: The number of hidden units for the BiLSTM layer.
        :param lstm_num_layers: The number of LSTM layers.
        :param lstm_dropout: The dropout rate for the BiLSTM layer.
        """
        super().__init__()

        text_dim = 769
        client_dim = 674
        self.linear_client = LinearLayer(dim * 2, dim)
        # 200 + 200 + 600 = 1000
        self.linear_preclasifier = LinearLayer(dim * 3, dim)
        self.bilstm_RoBERTa = BiLSTMLayer(
            input_size=text_dim,
            hidden_size=dim,
            num_layers=1,
        )
        self.bilstm_client = BiLSTMLayer(
            input_size=client_dim,
            hidden_size=dim,
            num_layers=1,
        )
        self.classifier = ClassiferLayer(dim, 2)

    def forward(self, x1, x2, x3: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        # unpack inputs or write forward function accepting 3 inputs
        x_RoBERTa, x_client, x_counselor = x1, x2, x3
        # RoBERTa
        # print(x_RoBERTa.shape)
        x_RoBERTa = self.bilstm_RoBERTa(x_RoBERTa)[0]
        x_RoBERTa = x_RoBERTa.flatten(start_dim=1)
        x_client = self.bilstm_client(x_client)[0]
        x_client = x_client.flatten(start_dim=1)
        # linear layers
        x_client = self.linear_client(x_client)
        # print(x_client.shape)
        # concantenate and linear layer
        x = torch.cat((x_RoBERTa, x_client), dim=1)
        x = self.linear_preclasifier(x)
        # classifier
        return self.classifier(x)

# used for pretrained weights, as the second stream
class CrossAttentionModel(nn.Module):
    """ 
    cross attention from client to text,
    cross attention from counseller to text,
    """
    def __init__(self, dim = 512, depth = 3, num_heads = 8, mlp_ratio = 4, 
                 qkv_bias = False, drop = 0, attn_drop = 0, init_values = 1e-5, 
                 drop_path = False, act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_len = 100
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        pe = positional_embedding(self.max_len, dim)  # [max_len, dim]
        self.register_buffer('pos_embed', pe)  # torch.Tensor, not a Parameter

        
        text_dim = 769
        client_dim = 674
        counseller_dim = 674

        # unify the dim of the input features
        self.linear_text = nn.Linear(text_dim, dim)
        self.linear_client = nn.Linear(client_dim, dim)
        self.linear_counseller = nn.Linear(counseller_dim, dim)
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

        self.final_norm = norm_layer(dim)
        self.classifier = ClassiferLayer(dim, 2, ifsoftmax=False, act_layer=act_layer, drop=drop)

    def forward(self, x, y, z):
        # positional embedding, may cause device error
        _, N, _ = x.shape
        assert N+1 <= self.max_len, f"Input too long: {N + 1} > {self.max_len}"

        x = self.linear_text(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1)
        # x = x + positional_embedding(N + 1, self.dim).to(x.device)
        # x = x + self.pos_embed[:N+1].to(x.device) 
        # _, M, _ = y.shape
        # y = self.linear_client(y) + self.pos_embed[:M].to(y.device)
        # y = self.linear_client(y) + positional_embedding(M, self.dim).to(y.device)
        # _, O, _ = z.shape
        # z = self.linear_counseller(z) + self.pos_embed[:O].to(z.device)
        # z = self.linear_counseller(z) + positional_embedding(O, self.dim).to(z.device)
        
        # for block in self.cl_blocks:
        #     x = block(x, y)
        # for block in self.co_blocks:
        #     x = block(x, z)
        # x = self.sT(x)
        for block in self.self_blocks:
            x = block(x)
        # x = x.flatten(start_dim=-2)
        # x = self.final_norm(x)
        x = x[:,0]
        print(x[0])
        return self.classifier(x)
    

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
                 drop_path = False, act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False, 
                 load_RoBERTa = False, freeze_RoBERTa = False):
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

        if load_RoBERTa:
            self.load_RoBERTa_weights()
            if freeze_RoBERTa:
                for param in self.bilstm_RoBERTa.parameters():
                    param.requires_grad = False

        
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

    def load_RoBERTa_weights(self, weights_path = None, map_location=None):
        """
        Load RoBERTa weights from a checkpoint.
        :param check_point_path: The path to the checkpoint.
        """
        # default/testing path when weights_path is not given.
        if weights_path is None:
            pth_name = "Roberta_only.pth"
            # local debugging path
            # weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "checkpoints", pth_name)
            
            # relative path from root direction: where python src/train.py is called.
            weights_path = os.path.join("data", "checkpoints", pth_name)
        
        ReBERTa_layers = ['bilstm']
        # default map location
        if map_location is None:
            map_location = 'cpu'
        state_dict = torch.load(weights_path, weights_only=True, map_location=map_location)
        filtered_state_dict = {k: v for k, v in state_dict.items() if any(k.startswith(prefix) for prefix in ReBERTa_layers)}
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        
        # debugging
        if True:
            print("\nLoaded Layers:")
            for k in filtered_state_dict.keys():
                print(f"  - {k}")

            if missing_keys:
                print("\nMissing Layers (not loaded from checkpoint, randomly initialized):")
                for k in missing_keys:
                    print(f"  - {k}")

            if unexpected_keys:
                print("\nUnexpected keys (checkpoint has these, but model doesn't expect them):")
                for k in unexpected_keys:
                    print(f"  - {k}")

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

# todo
class LSTMT_Interaction_Model(nn.Module):
    """ 
    LSTM on text, take hidden state as stream 1,
    Interaction refers to the cross attention between client and counseller and vice versa.
    Instead of a residual connection, the output of the cross attention is taken as the interactoins output.
    """
    def __init__(self, dim = 512, depth = 1, num_heads = 8, mlp_ratio = 4, 
                 qkv_bias = False, drop = 0, attn_drop = 0, init_values = 1e-5, 
                 drop_path = False, act_layer = nn.ReLU, norm_layer = nn.LayerNorm, causal = False, 
                 load_RoBERTa = False, freeze_RoBERTa = False):
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

        if load_RoBERTa:
            self.load_RoBERTa_weights()
            if freeze_RoBERTa:
                for param in self.bilstm_RoBERTa.parameters():
                    param.requires_grad = False

        
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

    def load_RoBERTa_weights(self, weights_path = None, map_location=None):
        """
        Load RoBERTa weights from a checkpoint.
        :param check_point_path: The path to the checkpoint.
        """
        # default/testing path when weights_path is not given.
        if weights_path is None:
            pth_name = "Roberta_only.pth"
            # local debugging path
            # weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "checkpoints", pth_name)
            
            # relative path from root direction: where python src/train.py is called.
            weights_path = os.path.join("data", "checkpoints", pth_name)
        
        ReBERTa_layers = ['bilstm']
        # default map location
        if map_location is None:
            map_location = 'cpu'
        state_dict = torch.load(weights_path, weights_only=True, map_location=map_location)
        filtered_state_dict = {k: v for k, v in state_dict.items() if any(k.startswith(prefix) for prefix in ReBERTa_layers)}
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        
        # debugging
        if True:
            print("\nLoaded Layers:")
            for k in filtered_state_dict.keys():
                print(f"  - {k}")

            if missing_keys:
                print("\nMissing Layers (not loaded from checkpoint, randomly initialized):")
                for k in missing_keys:
                    print(f"  - {k}")

            if unexpected_keys:
                print("\nUnexpected keys (checkpoint has these, but model doesn't expect them):")
                for k in unexpected_keys:
                    print(f"  - {k}")

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
    import os
    a = torch.rand((5,5,769))
    b = torch.rand((5,6,674))
    # pth_name = "Roberta_only.pth"
    # state_dict_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "checkpoints", pth_name)
    # print(state_dict_path)
    # state_dict = torch.load(state_dict_path, weights_only= True, map_location='cpu')

    # allowed_prefixes = ['bilstm']
    # filtered_state_dict = {k: v for k, v in state_dict.items() if any(k.startswith(prefix) for prefix in allowed_prefixes)}
    # model = LSTMT_cCl_cCo_Model()

    # missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    # print("\nLoaded Layers:")
    # for k in filtered_state_dict.keys():
    #     print(f"  - {k}")

    # if missing_keys:
    #     print("\nMissing Layers (not loaded from checkpoint, randomly initialized):")
    #     for k in missing_keys:
    #         print(f"  - {k}")

    # if unexpected_keys:
    #     print("\nUnexpected keys (checkpoint has these, but model doesn't expect them):")
    #     for k in unexpected_keys:
    #         print(f"  - {k}")

    model = LSTMT_lstmT_cCl_cCo_Model(dim=512, load_RoBERTa=True, freeze_RoBERTa=True)
    output = model(a, b, b)
    print(output.shape)
