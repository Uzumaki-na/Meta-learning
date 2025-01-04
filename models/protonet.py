import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from einops import rearrange

class ProtoNetEmbedding(nn.Module):
    def __init__(self, input_dim: int = 1, input_size: int = 28, embedding_dim: int = 64, transformer_layers: int = 2, transformer_heads: int = 8, transformer_dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            self._conv_block(input_dim, 64),
            self._conv_block(64, 64),
            self._conv_block(64, embedding_dim)
        )
        with torch.no_grad():
            test_input = torch.zeros(1, input_dim, input_size, input_size)
            features = self.encoder(test_input)
            flattened_size = features.view(1, -1).size(1)
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        self.transformer = TransformerEncoder(
            embedding_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=transformer_dropout
        )
    
    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        if x.size(2) != self.input_size or x.size(3) != self.input_size:
            x = F.interpolate(x, (self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.transformer(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class PrototypicalNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_network = ProtoNetEmbedding(
            input_dim=1,
            input_size=config.INPUT_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_dropout=config.TRANSFORMER_DROPOUT
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_network(x)
    
    def compute_prototypes(self, support_images: torch.Tensor, 
                          support_labels: torch.Tensor) -> torch.Tensor:
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = support_labels == label
            class_embeddings = self.embedding_network(support_images[mask])
            prototype = class_embeddings.mean(0)
            prototypes.append(prototype)
            
        return torch.stack(prototypes)
    
    def compute_distances(self, query_embeddings: torch.Tensor, 
                         prototypes: torch.Tensor) -> torch.Tensor:
        n_queries = query_embeddings.size(0)
        n_prototypes = prototypes.size(0)
        
        query_embeddings = query_embeddings.unsqueeze(1).expand(-1, n_prototypes, -1)
        prototypes = prototypes.unsqueeze(0).expand(n_queries, -1, -1)
        
        return torch.pow(query_embeddings - prototypes, 2).sum(2)
    
    def compute_loss(self, query_embeddings: torch.Tensor, 
                    query_labels: torch.Tensor, 
                    prototypes: torch.Tensor,
                    label_smoothing:float = 0.0) -> torch.Tensor:
        distances = self.compute_distances(query_embeddings, prototypes)
        log_p_y = F.log_softmax(-distances, dim=1)
        if label_smoothing > 0.0:
            loss = self._label_smoothing_loss(log_p_y, query_labels, label_smoothing)
        else:
            loss = F.nll_loss(log_p_y, query_labels)
        return loss, log_p_y
    
    def _label_smoothing_loss(self, log_probs, target, smoothing):
        confidence = 1.0 - smoothing
        smoothed_probs = torch.full_like(log_probs, smoothing / log_probs.size(1))
        smoothed_probs.scatter_(1, target.unsqueeze(1), confidence)
        loss = -torch.sum(smoothed_probs * log_probs, dim=1)
        return loss.mean()