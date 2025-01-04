import torch
import os
from typing import Optional
import yaml

class Config:
    def __init__(self, config_file='config.yaml'):
        self.config_file = config_file
        self._load_config()
        self._initialize_directories()
    
    def _load_config(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.DATASET_PATH = config.get('DATASET_PATH', './data/omniglot')
        self.NUM_CLASSES = config.get('NUM_CLASSES', 1623)
        
        self.N_WAY = config.get('N_WAY', 5)
        self.K_SHOT = config.get('K_SHOT', 1)
        self.Q_QUERY = config.get('Q_QUERY', 15)
        
        self.EMBEDDING_DIM = config.get('EMBEDDING_DIM', 128)
        self.INPUT_SIZE = config.get('INPUT_SIZE', 28)
        self.TRANSFORMER_LAYERS = config.get('TRANSFORMER_LAYERS', 2)
        self.TRANSFORMER_HEADS = config.get('TRANSFORMER_HEADS', 8)
        self.TRANSFORMER_DROPOUT = float(config.get('TRANSFORMER_DROPOUT', 0.1)) # Cast to float
        
        self.EPOCHS = config.get('EPOCHS', 100)
        self.LEARNING_RATE = float(config.get('LEARNING_RATE', 1e-3)) #Cast to float
        self.BATCH_SIZE = config.get('BATCH_SIZE', 32)
        self.WEIGHT_DECAY = float(config.get('WEIGHT_DECAY', 1e-4)) #Cast to float
        self.GRADIENT_ACCUMULATION_STEPS = config.get('GRADIENT_ACCUMULATION_STEPS', 1)
        self.LABEL_SMOOTHING = float(config.get('LABEL_SMOOTHING', 0.1)) #Cast to float
        
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.USE_MIXED_PRECISION = config.get('USE_MIXED_PRECISION', True) and torch.cuda.is_available()
        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.CHECKPOINT_DIR = config.get('CHECKPOINT_DIR', os.path.join(self.BASE_DIR, 'checkpoints'))
        self.LOGS_DIR = config.get('LOGS_DIR', os.path.join(self.BASE_DIR, 'logs'))
        self.CHECKPOINT_PATH = config.get('CHECKPOINT_PATH', os.path.join(self.CHECKPOINT_DIR, 'best_model.pth'))
        self.TENSORBOARD_LOG_DIR = config.get('TENSORBOARD_LOG_DIR', self.LOGS_DIR)
        
        self.USE_DATA_AUGMENTATION = config.get('USE_DATA_AUGMENTATION', False)
        self.USE_LR_SCHEDULER = config.get('USE_LR_SCHEDULER', False)
        self.USE_TENSORBOARD = config.get('USE_TENSORBOARD', True)
        self.USE_GRADIENT_CLIPPING = config.get('USE_GRADIENT_CLIPPING', True)
        self.GRADIENT_CLIP_VALUE = float(config.get('GRADIENT_CLIP_VALUE', 1.0)) #Cast to float
        self.USE_EARLY_STOPPING = config.get('USE_EARLY_STOPPING', True)
        self.EARLY_STOPPING_PATIENCE = config.get('EARLY_STOPPING_PATIENCE', 20)
        self.USE_WANDB = config.get('USE_WANDB', False)
        self.SEED = config.get('SEED', 42)
        self.OPTIMIZER = config.get('OPTIMIZER', 'adamw')
        self.META_LEARNING_ALGO = config.get('META_LEARNING_ALGO', 'protonet')
    
    def _initialize_directories(self) -> None:
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
    
    def save_config(self, path):
      with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)