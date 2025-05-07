import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.checkpoint
# from src.models.components.model_helper import 

def positional_embedding(n, dim):
    pe = torch.zeros(n, dim)
    position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

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
