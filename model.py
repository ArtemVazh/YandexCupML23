import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_dim=768, mult=4, p=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mult),
            nn.Dropout(p),
            nn.GELU(),
            nn.Linear(emb_dim * mult, emb_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
class AttentionPooling(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 1)
        )

    def forward(self, x, mask=None):
        attn_logits = self.attn(x)
        if mask is not None:
            attn_logits[mask] = -float('inf')
        attn_weights = torch.softmax(attn_logits, dim=1)
        x = x * attn_weights
        x = x.sum(dim=1)
        return x
    
class Network(nn.Module):
    def __init__(
        self,
        num_classes = 256,
        input_dim = 768,
        hidden_dim = 512,
        nhead = 12,
        num_layers = 6,
        lstm = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.proj = FeedForward(input_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.poooling = AttentionPooling(input_dim)
        self.fc = nn.Linear(input_dim, num_classes)
               
    def forward(self, embeds):
        embeds = self.proj(embeds)
        src_key_padding_mask = (embeds.mean(-1) == -1)
        embeds = self.ln(embeds)
        x = self.transformer_encoder(embeds, src_key_padding_mask=src_key_padding_mask)
        x = self.bn(self.poooling(x, mask=src_key_padding_mask))
        outs = self.fc(x)
        return outs