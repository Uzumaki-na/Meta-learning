import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler
import wandb
from config import Config  

def train_one_epoch(model: nn.Module, 
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    config: Config,
                    writer: Optional[object] = None,
                    epoch: int = 0) -> dict:
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True) # Progress bar
    
    for batch_idx, batch in progress_bar:
        support_images, support_labels, query_images, query_labels = [
            item.to(config.DEVICE) for item in batch
        ]
        
        query_embeddings = model(query_images)
        prototypes = model.compute_prototypes(support_images, support_labels)
        
        loss, log_probabilities = model.compute_loss(query_embeddings, query_labels, prototypes, label_smoothing=config.LABEL_SMOOTHING)
        predictions = log_probabilities.argmax(dim=1)
        
        correct_predictions += (predictions == query_labels).sum().item()
        total_predictions += query_labels.size(0)
        
        loss.backward()
          
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or batch_idx == len(data_loader) - 1:
            if config.USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config.GRADIENT_CLIP_VALUE
                    )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0 
        if isinstance(loss.item(), float):
          progress_bar.set_postfix({'loss': loss.item(), 'accuracy': accuracy, "predictions": predictions.tolist(), "labels": query_labels.tolist(), "support_labels": support_labels.tolist()})
        else:
           progress_bar.set_postfix({'accuracy': accuracy, "predictions": predictions.tolist(), "labels": query_labels.tolist(), "support_labels": support_labels.tolist()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    
    if writer:
        writer.add_scalar('Training/Loss', avg_loss, epoch)
        writer.add_scalar('Training/Accuracy', accuracy, epoch)
    
    return {'loss': avg_loss, 'accuracy': accuracy}

def train_maml_one_epoch(model: nn.Module, 
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    config: Config,
                    writer: Optional[object] = None,
                    epoch: int = 0) -> dict:
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True) # Progress bar
    
    for batch_idx, batch in progress_bar:
        support_images, support_labels, query_images, query_labels = [
            item.to(config.DEVICE) for item in batch
        ]
        
        loss, log_probabilities = model.compute_loss(support_images, support_labels, query_images, query_labels)
        predictions = log_probabilities.argmax(dim=1)
        
        correct_predictions += (predictions == query_labels).sum().item()
        total_predictions += query_labels.size(0)
        
        loss.backward()
          
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or batch_idx == len(data_loader) - 1:
            if config.USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config.GRADIENT_CLIP_VALUE
                    )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0 # prevent division by 0
        if isinstance(loss.item(), float):
          progress_bar.set_postfix({'loss': loss.item(), 'accuracy': accuracy, "predictions": predictions.tolist(), "labels": query_labels.tolist(), "support_labels": support_labels.tolist()})
        else:
           progress_bar.set_postfix({'accuracy': accuracy, "predictions": predictions.tolist(), "labels": query_labels.tolist(), "support_labels": support_labels.tolist()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    
    if writer:
        writer.add_scalar('Training/Loss', avg_loss, epoch)
        writer.add_scalar('Training/Accuracy', accuracy, epoch)
    
    return {'loss': avg_loss, 'accuracy': accuracy}

def train_reptile_one_epoch(model: nn.Module, 
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    config: Config,
                    writer: Optional[object] = None,
                    epoch: int = 0) -> dict:
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True) # Progress bar
    
    for batch_idx, batch in progress_bar:
        support_images, support_labels, query_images, query_labels = [
            item.to(config.DEVICE) for item in batch
        ]
        
        loss, log_probabilities = model.compute_loss(support_images, support_labels, query_images, query_labels)
        predictions = log_probabilities.argmax(dim=1)
        
        correct_predictions += (predictions == query_labels).sum().item()
        total_predictions += query_labels.size(0)
        
        loss.backward()
          
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or batch_idx == len(data_loader) - 1:
            if config.USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config.GRADIENT_CLIP_VALUE
                    )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0 
        if isinstance(loss.item(), float):
          progress_bar.set_postfix({'loss': loss.item(), 'accuracy': accuracy, "predictions": predictions.tolist(), "labels": query_labels.tolist(), "support_labels": support_labels.tolist()})
        else:
           progress_bar.set_postfix({'accuracy': accuracy, "predictions": predictions.tolist(), "labels": query_labels.tolist(), "support_labels": support_labels.tolist()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    
    if writer:
        writer.add_scalar('Training/Loss', avg_loss, epoch)
        writer.add_scalar('Training/Accuracy', accuracy, epoch)
    
    return {'loss': avg_loss, 'accuracy': accuracy}


def visualize_training_progress(train_histories, path=None):

    plt.figure(figsize=(12, 5))
    
    if isinstance(train_histories, dict):
        plt.subplot(1, 2, 1)
        for algo, history in train_histories.items():
            plt.plot(history['losses'], label=f'{algo} Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for algo, history in train_histories.items():
            plt.plot(history['accuracies'], label=f'{algo} Accuracy')
        plt.title('Training Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    # If input is a list of losses and accuracies
    elif isinstance(train_histories, list):
        plt.plot(train_histories, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    # If input is a tuple of losses and accuracies
    elif isinstance(train_histories, tuple) and len(train_histories) == 2:
        losses, accuracies = train_histories
        
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Training Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    else:
        raise ValueError("Invalid input format for visualize_training_progress")
    
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
    
def learning_rate_scheduler(optimizer, epoch, initial_lr, decay_rate=0.1, decay_epochs=30):
    lr = initial_lr * (decay_rate ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def get_lr_scheduler(optimizer, config):
  if config.USE_LR_SCHEDULER == 'plateau':
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
  elif config.USE_LR_SCHEDULER == 'cosine':
      return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=config.LEARNING_RATE * 0.1, verbose=True)
  elif config.USE_LR_SCHEDULER == 'onecycle':
        return OneCycleLR(optimizer, max_lr=config.LEARNING_RATE, epochs=config.EPOCHS, steps_per_epoch=1, pct_start=0.3, div_factor=25.0, final_div_factor=1000.0)
  else:
      return None

def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    return torch.pow(x - y, 2).sum(2)