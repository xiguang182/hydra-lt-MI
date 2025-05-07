import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
# from src.models.components.model_helper import 

def positional_embedding(n, dim):
    pe = torch.zeros(n, dim)
    position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

    
class CrossAttention(nn.Module):
    """
    Causal Cross-Attention Module that implements the functionality of nn.MultiheadAttention
    with explicit causal masking support.
    
    This implementation properly separates query input from key/value inputs
    for true cross-attention behavior.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.0)
        bias (bool): If True, use bias in the projection layers (default: True)
    """
    
    def __init__(self, embed_dim = 256, num_heads = 4, dropout=0.0, bias=True):
        super().__init__()
        
        # Validate parameters
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # Scaling factor for dot product attention
        self.scale = self.head_dim ** -0.5
        
        # Separate projection matrices for Q, K, V
        # Q comes from target sequence, K and V come from source sequence
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the weights similar to PyTorch's default initialization."""
        # Xavier uniform initialization helps maintain variance across layers
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Initialize biases to zero if they exist
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, query, key, value, causal=False, need_weights=False, key_padding_mask=None, attn_mask=None):
        """
        Forward pass of the causal cross-attention module.
        
        Args:
            query (Tensor): Query tensor from target sequence
            key (Tensor): Key tensor from source sequence
            value (Tensor): Value tensor from source sequence
            causal (bool, optional): If True, applies causal masking (default: False)
            need_weights (bool, optional): If True, returns attention weights
            key_padding_mask (Tensor, optional): Mask for padding in key sequence
            attn_mask (Tensor, optional): Additional attention mask

        Returns:
            Tensor: Output tensor
            Tensor (optional): Attention weights if need_weights=True
        """
               
        # Get dimensions
        bsz, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        
        # Apply projections to separate inputs
        # This is proper cross-attention: q from query sequence, k/v from key/value sequences
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = q.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.contiguous().view(bsz, src_len, self.num_heads, self.head_dim)
        v = v.contiguous().view(bsz, src_len, self.num_heads, self.head_dim)
        
        # Transpose for batched matrix multiplication
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, num_heads, head_dim, seq_len]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        # [batch_size, num_heads, tgt_len, head_dim] @ [batch_size, num_heads, head_dim, src_len]
        # -> [batch_size, num_heads, tgt_len, src_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create causal mask and apply it
        # The mask should prevent positions from attending to subsequent positions
        if causal:
            # Create a mask with shape [N, N]
            # Masking future positions (upper triangle of the matrix)
            # to device?
            causal_mask = torch.triu(torch.ones(tgt_len, src_len, dtype=torch.bool, device=attn_weights.device), diagonal=1)

            # Expand causal_mask for batch size and num_heads
            # [tgt_len, src_len] -> [1, 1, tgt_len, src_len] -> broadcast to batch and heads
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            # Apply the causal mask: setting masked positions to -inf
            attn_weights.masked_fill_(causal_mask, float('-inf'))

        # Apply additional attention mask if provided
        if attn_mask is not None:
            # Ensure the mask has the right shape based on its dimensionality
            if attn_mask.dim() == 2:
                # [tgt_len, src_len] -> [1, 1, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [batch_size, tgt_len, src_len] -> [batch_size, 1, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(1)
            
            attn_weights = attn_weights + attn_mask  # Additive mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # [batch_size, src_len] -> [batch_size, 1, 1, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))


        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention weights to values
        # [batch_size, num_heads, tgt_len, src_len] @ [batch_size, num_heads, src_len, head_dim]
        # -> [batch_size, num_heads, tgt_len, head_dim]
        output = torch.matmul(attn_weights, v)
        
        # Transpose and reshape the output
        # [batch_size, num_heads, tgt_len, head_dim] -> [batch_size, tgt_len, num_heads, head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output, attn_weights if need_weights else None


class CA_model(nn.Module):
    def __init__(self, RoBETa_dim=769, openFace_dim=674,  model_dim=256, num_heads=4, num_classes=2, dropout=0.1):
        super().__init__()
        
        self.model_dim = model_dim
        self.max_len = 512  # Maximum sequence length
        pe = positional_embedding(self.max_len, model_dim)  # [max_len, dim]
        self.register_buffer('pos_embed', pe)  # torch.Tensor, not a Parameter

        
        # Project RoBERTa to model dimension
        self.RoBERTa_proj = nn.Linear(RoBETa_dim, model_dim)
        self.openface_proj = nn.Linear(openFace_dim, model_dim)
        # pre-atten norm
        self.pre_attn_norm_x = nn.LayerNorm(model_dim)
        self.pre_attn_norm_y = nn.LayerNorm(model_dim)
        # Multi-head self-attention
        self.attn = CrossAttention(embed_dim=model_dim, num_heads=num_heads)
        
        # Post-attention processing
        ## pre-MLP norm
        self.pre_mlp_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )

        # Final classifier
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x, y, z):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        y: Tensor of shape (batch_size, seq_len, input_dim)
        z: unused, for lightning model compatibility
        """
        # Project input
        x_proj = self.RoBERTa_proj(x)
        y_proj = self.openface_proj(y)
        # (B, N, model_dim)
        B, N, _ = x_proj.shape
        # Add positional embedding
        x_proj = x_proj + self.pos_embed[:N, :].unsqueeze(0)
        y_proj = y_proj + self.pos_embed[:N, :].unsqueeze(0)
        # Self-attention
        x_proj = self.pre_attn_norm_x(x_proj)
        y_proj = self.pre_attn_norm_y(y_proj)
        attn_out, _ = self.attn(query=x_proj, key=y_proj, value=y_proj) 
        # Residual + dropout
        x_res = x_proj + self.dropout(attn_out)
        x_res = self.pre_mlp_norm(x_res)
        x_mlp = x_res + self.dropout(self.mlp(x_res))
        
        pooled = x_mlp.mean(dim=1)
        # Classify
        logits = self.classifier(pooled)
        # (B, num_classes)
        return logits
        


# test model for debugging
class SA_RoBERTa(nn.Module):
    def __init__(self, input_dim=769, model_dim=256, num_heads=4, num_classes=2, dropout=0.1, ifcls_token=False):
        super().__init__()
        
        self.model_dim = model_dim
        self.ifcls_token = ifcls_token
        if ifcls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.max_len = 512  # Maximum sequence length
        pe = positional_embedding(self.max_len, model_dim)  # [max_len, dim]
        self.register_buffer('pos_embed', pe)  # torch.Tensor, not a Parameter

        
        # Project RoBERTa to model dimension
        self.RoBERTa_proj = nn.Linear(input_dim, model_dim)
        # pre-atten norm
        self.norm1 = nn.LayerNorm(model_dim)
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        
        # Post-attention processing
        ## pre-MLP norm
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )

        # Final classifier
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x, y, z):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # Project input
        x_proj = self.RoBERTa_proj(x)  # (B, T, model_dim)

        B, N, _ = x_proj.shape
        # Add CLS token
        if self.ifcls_token:
            x_proj = torch.cat((self.cls_token.expand(B, -1, -1), x_proj), dim = 1)  # (B, N+1, model_dim)
            N += 1
        # Add positional embedding
        x_proj = x_proj + self.pos_embed[:N, :].unsqueeze(0)  # (B, T, model_dim)
        
        
        # Self-attention
        x_proj = self.norm1(x_proj)  # pre-atten norm
        attn_out, _ = self.attn(x_proj, x_proj, x_proj)  # (B, T, model_dim)
        
        # Residual + dropout
        x_res = x_proj + self.dropout(attn_out)

        x_res = self.norm2(x_res) # pre-MLP norm

        x_mlp = x_res + self.dropout(self.mlp(x_res))  # (B, T, model_dim)

        # Mean pooling across tokens
        # pooled = x_mlp.mean(dim=1)  # (B, model_dim)
        if self.ifcls_token:
            # CLS token
            pooled = x_mlp[:, 0, :]
        else:
            # Mean pooling
            pooled = x_mlp.mean(dim=1)

        # Classify
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits
    

if __name__ == "__main__":
    pass
