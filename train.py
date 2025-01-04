import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import random
import os
import argparse

from config import Config
from models.protonet import PrototypicalNetwork
from models.maml import MAML
from models.reptile import Reptile
from data.omniglot_download import OmniglotDownloader
from utils.data_loader import OmniglotDataset, create_data_loader, get_default_transform
from utils.training_utils import train_one_epoch, visualize_training_progress, get_lr_scheduler, train_maml_one_epoch, train_reptile_one_epoch
from torch.cuda.amp import GradScaler
import wandb

class Trainer:
    def __init__(self, config: Config, meta_learning_algo: str = None):
        self.config = config
        self.device = config.DEVICE
        self.meta_learning_algo = meta_learning_algo
        self.initialize_components()
        
    def initialize_components(self) -> None:
        """Initialize all training components"""
        Config()._initialize_directories()
        self._set_seed(self.config.SEED)
        
        self._setup_dataset()
        
        if self.meta_learning_algo == 'protonet':
            self.model = PrototypicalNetwork(self.config).to(self.device)
        elif self.meta_learning_algo == 'maml':
             self.model = MAML(self.config).to(self.device)
        elif self.meta_learning_algo == 'reptile':
            self.model = Reptile(self.config).to(self.device)
        elif self.meta_learning_algo is None:
           pass # Handled by train function
        else:
            raise ValueError(f"Meta learning algorithm '{self.meta_learning_algo}' not supported")
        
        if self.meta_learning_algo is not None:
           print(f"Training {self.meta_learning_algo} on: {self.device}")
           
           if self.config.OPTIMIZER == 'adamw':
             self.optimizer = torch.optim.AdamW(
               self.model.parameters(),
               lr=self.config.LEARNING_RATE,
               weight_decay=self.config.WEIGHT_DECAY
            )
           elif self.config.OPTIMIZER == 'sgd':
               self.optimizer = torch.optim.SGD(
                 self.model.parameters(),
                 lr=self.config.LEARNING_RATE,
                 weight_decay = self.config.WEIGHT_DECAY,
                 momentum=0.9
            )
           else:
               raise ValueError(f"Optimizer '{self.config.OPTIMIZER}' not supported")
        
           self.scheduler = get_lr_scheduler(self.optimizer, self.config)
           
        self.writer = SummaryWriter(self.config.TENSORBOARD_LOG_DIR) if self.config.USE_TENSORBOARD else None
        
        self.scaler = GradScaler(enabled=self.config.USE_MIXED_PRECISION)
        
        if self.config.USE_WANDB:
          wandb.init(project="few-shot-omniglot", config=self.config.__dict__)
    
    def _set_seed(self, seed):
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
        
    def _setup_dataset(self) -> None:
        """Setup dataset and dataloader"""
        dataset_path = OmniglotDownloader.download_dataset(self.config.DATASET_PATH)
        transform = get_default_transform(self.config)
        self.dataset = OmniglotDataset(root_dir=dataset_path, transform=transform)
        self.data_loader = create_data_loader(
            self.dataset,
            self.config.N_WAY,
            self.config.K_SHOT,
            self.config.Q_QUERY,
            self.config.BATCH_SIZE
        )
    
    def _validate_configuration(self):
      if self.config.USE_MIXED_PRECISION and not torch.cuda.is_available():
        print("Warning: Mixed precision training is enabled but CUDA is not available. Disabling mixed precision.")
        self.config.USE_MIXED_PRECISION = False
      if self.config.USE_WANDB and "WANDB_API_KEY" not in os.environ:
          print("Warning: wandb is enabled but no API key set. Disabling wandb.")
          self.config.USE_WANDB = False
    
    def train(self) -> Dict[str, float]:
        """Execute training loop"""
        self._validate_configuration()
        
        if self.meta_learning_algo is None:
            # Train all models back-to-back if no model is specified
            all_algos = ['protonet', 'maml', 'reptile']
            all_training_histories = {}
            for algo in all_algos:
               print(f"\nTraining {algo} model:")
               self.meta_learning_algo = algo
               self.initialize_components()
               best_loss = float('inf')
               training_history = {'losses': [], 'accuracies': []}
               early_stopping_counter = 0
               for epoch in range(self.config.EPOCHS):
                  # Train one epoch
                  if self.meta_learning_algo == 'protonet':
                      metrics = train_one_epoch(
                          self.model,
                          self.data_loader,
                          self.optimizer,
                          self.config,
                          self.writer,
                          epoch
                      )
                  elif self.meta_learning_algo == 'maml':
                      metrics = train_maml_one_epoch(
                          self.model,
                          self.data_loader,
                          self.optimizer,
                          self.config,
                          self.writer,
                          epoch
                      )
                  elif self.meta_learning_algo == 'reptile':
                      metrics = train_reptile_one_epoch(
                        self.model,
                          self.data_loader,
                          self.optimizer,
                          self.config,
                          self.writer,
                          epoch
                    )
                  else:
                      raise ValueError(f"Meta learning algorithm '{self.meta_learning_algo}' not supported")
        
                  training_history['losses'].append(metrics['loss'])
                  training_history['accuracies'].append(metrics['accuracy'])
            
                  print(f"Epoch {epoch+1}/{self.config.EPOCHS} - "
                    f"Loss: {metrics['loss']:.4f} - "
                      f"Accuracy: {metrics['accuracy']:.4f}")
                  if self.config.USE_WANDB:
                    wandb.log({
                     "epoch": epoch,
                      "loss": metrics['loss'],
                     "accuracy": metrics['accuracy']
                      })
            
                  if self.config.USE_LR_SCHEDULER and self.scheduler:
                    if self.config.USE_LR_SCHEDULER == 'onecycle':
                        self.scheduler.step()
                    else:
                        self.scheduler.step(metrics['loss'])
            
                  if metrics['loss'] < best_loss:
                     best_loss = metrics['loss']
                     self.save_model()
                     early_stopping_counter = 0
                  elif self.config.USE_EARLY_STOPPING:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        print("Early stopping triggered")
                        break
               all_training_histories[algo] = training_history
               if self.writer:
                   self.writer.close()
            visualize_training_progress(all_training_histories, Path(self.config.LOGS_DIR) / 'training_history.png')
            if self.config.USE_WANDB:
               wandb.finish()
            return all_training_histories
        else:
          best_loss = float('inf')
          training_history = {'losses': [], 'accuracies': []}
          early_stopping_counter = 0
          for epoch in range(self.config.EPOCHS):
            if self.meta_learning_algo == 'protonet':
                metrics = train_one_epoch(
                    self.model,
                    self.data_loader,
                    self.optimizer,
                    self.config,
                    self.writer,
                    epoch
                )
            elif self.meta_learning_algo == 'maml':
               metrics = train_maml_one_epoch(
                   self.model,
                   self.data_loader,
                   self.optimizer,
                   self.config,
                   self.writer,
                   epoch
               )
            elif self.meta_learning_algo == 'reptile':
                metrics = train_reptile_one_epoch(
                    self.model,
                    self.data_loader,
                    self.optimizer,
                    self.config,
                    self.writer,
                    epoch
                )
            else:
              raise ValueError(f"Meta learning algorithm '{self.meta_learning_algo}' not supported")
            
            training_history['losses'].append(metrics['loss'])
            training_history['accuracies'].append(metrics['accuracy'])
            
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} - "
                  f"Loss: {metrics['loss']:.4f} - "
                  f"Accuracy: {metrics['accuracy']:.4f}")
            if self.config.USE_WANDB:
              wandb.log({
                  "epoch": epoch,
                  "loss": metrics['loss'],
                  "accuracy": metrics['accuracy']
              })
            
            if self.config.USE_LR_SCHEDULER and self.scheduler:
                if self.config.USE_LR_SCHEDULER == 'onecycle':
                    self.scheduler.step()
                else:
                   self.scheduler.step(metrics['loss'])
            
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                self.save_model()
                early_stopping_counter = 0
            elif self.config.USE_EARLY_STOPPING:
                early_stopping_counter += 1
                if early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                  print("Early stopping triggered")
                  break
          visualize_training_progress(training_history['losses'], training_history['accuracies'], Path(self.config.LOGS_DIR) / 'training_history.png')
          if self.writer:
            self.writer.close()
        
          if self.config.USE_WANDB:
            wandb.finish()
          return training_history
    
    def save_model(self) -> None:
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'best_model_{self.meta_learning_algo}.pth'
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

def main():
    config = Config()
    config.save_config(Path(config.LOGS_DIR) / 'config.yaml')
    
    parser = argparse.ArgumentParser(description="Train few-shot learning models.")
    parser.add_argument("--model", type=str, default=None, help="Specify which model to train (protonet, maml, reptile). If not provided, trains all three back-to-back.")

    args = parser.parse_args()
    trainer = Trainer(config, args.model)
    trainer.train()

if __name__ == '__main__':
    main()