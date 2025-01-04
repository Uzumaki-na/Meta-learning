import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple

from models.protonet import ProtoNetEmbedding
from config import Config

class Reptile(nn.Module):
    def __init__(self, config: Config):
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
        self.meta_lr = config.LEARNING_RATE
        self.inner_lr = config.LEARNING_RATE * 0.1
        self.task_grad_steps = 5 
        
    def forward(self, x):
        return self.embedding_network(x)
        
    def _inner_loop(self, support_images, support_labels):
      fast_weights = list(self.embedding_network.parameters())
      
      for _ in range(self.task_grad_steps):
            #Compute support set embeddings
            support_embeddings = self.embedding_network(support_images)
            prototypes = self._compute_prototypes(support_embeddings, support_labels)
            distances = self._compute_distances(support_embeddings, prototypes)
            log_p_y = F.log_softmax(-distances, dim=1)
            loss = F.nll_loss(log_p_y, support_labels)
          
            #Compute gradients with respect to parameters
            grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
          
            #Update the fast weights (in-place update)
            for w, g in zip(fast_weights, grad):
                w.data = w.data - self.inner_lr * g.data
      return fast_weights
    
    def _compute_prototypes(self, support_embeddings, support_labels):
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = support_labels == label
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(0)
            prototypes.append(prototype)
            
        return torch.stack(prototypes)
    
    def _compute_distances(self, query_embeddings: torch.Tensor, 
                         prototypes: torch.Tensor) -> torch.Tensor:
        n_queries = query_embeddings.size(0)
        n_prototypes = prototypes.size(0)
        
        query_embeddings = query_embeddings.unsqueeze(1).expand(-1, n_prototypes, -1)
        prototypes = prototypes.unsqueeze(0).expand(n_queries, -1, -1)
        
        return torch.pow(query_embeddings - prototypes, 2).sum(2)
    
    def compute_loss(self, support_images, support_labels, query_images, query_labels):
      fast_weights = self._inner_loop(support_images, support_labels)
      query_embeddings = self.embedding_network(query_images)
      prototypes = self._compute_prototypes(self.embedding_network(support_images), support_labels)
      distances = self._compute_distances(query_embeddings, prototypes)
      log_p_y = F.log_softmax(-distances, dim=1)
      loss = F.nll_loss(log_p_y, query_labels)
      return loss, log_p_y
    
    def test(self, support_images, support_labels, query_images):
        fast_weights = self._inner_loop(support_images, support_labels)
        query_embeddings = self.embedding_network(query_images)
        prototypes = self._compute_prototypes(self.embedding_network(support_images), support_labels)
        distances = self._compute_distances(query_embeddings, prototypes)
        log_p_y = F.log_softmax(-distances, dim=1)
        predictions = log_p_y.argmax(dim=1)
        return predictions