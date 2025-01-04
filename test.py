import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import os
import yaml
from config import Config
from models.protonet import PrototypicalNetwork
from models.maml import MAML
from models.reptile import Reptile
from data.omniglot_download import OmniglotDownloader
from utils.data_loader import OmniglotDataset, create_episode, get_default_transform
from utils.training_utils import euclidean_distance
import wandb

class Tester:
    def __init__(self, config: Config, model_type: str = 'best'):
        self.config = config
        self.device = config.DEVICE
        self.model_type = model_type
        self.initialize_components()
    
    def initialize_components(self) -> None:
        """Initialize testing components"""
        self._setup_dataset()
        
        if self.model_type == 'best':
           best_model_metrics = {}
           best_model_type = None
           for model_type in ['protonet', 'maml', 'reptile']:
                try:
                    metrics_path = os.path.join(self.config.LOGS_DIR, f'metrics_evaluate_{model_type}.yaml')
                    with open(metrics_path, 'r') as f:
                         metrics = yaml.safe_load(f)
                    best_model_metrics[model_type] = metrics['accuracy']
                except FileNotFoundError:
                  best_model_metrics[model_type] = -1
           best_model_type = max(best_model_metrics, key = best_model_metrics.get)
           if best_model_metrics[best_model_type] == -1:
               raise ValueError("No models available for evaluation")
           self.config.META_LEARNING_ALGO = best_model_type
        elif self.model_type in ['protonet', 'maml', 'reptile']:
            self.config.META_LEARNING_ALGO = self.model_type
        else:
            raise ValueError(f"Model type {self.model_type} not recognised")
           
        if self.config.META_LEARNING_ALGO == 'protonet':
            self.model = PrototypicalNetwork(self.config)
        elif self.config.META_LEARNING_ALGO == 'maml':
            self.model = MAML(self.config)
        elif self.config.META_LEARNING_ALGO == 'reptile':
            self.model = Reptile(self.config)
        else:
            raise ValueError(f"Meta learning algorithm '{self.config.META_LEARNING_ALGO}' not supported")
        
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, f'best_model_{self.config.META_LEARNING_ALGO}.pth')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def _setup_dataset(self) -> None:
        dataset_path = OmniglotDownloader.download_dataset(
            self.config.DATASET_PATH,
            dataset_type='evaluation'
        )
        transform = get_default_transform(self.config)
        self.dataset = OmniglotDataset(
            root_dir=dataset_path,
            split='evaluation',
            transform=transform,
            preload_images=True
        )
    
    def test(self, num_episodes: int = 100) -> Dict[str, float]:
        all_predictions = []
        all_labels = []
        accuracies = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                support_images, support_labels, query_images, query_labels = create_episode(
                    self.dataset,
                    self.config.N_WAY,
                    self.config.K_SHOT,
                    self.config.Q_QUERY
                )
                
                support_images = support_images.to(self.device)
                support_labels = support_labels.to(self.device)
                query_images = query_images.to(self.device)
                query_labels = query_labels.to(self.device)
                
                if self.config.META_LEARNING_ALGO == 'protonet':
                  query_embeddings = self.model(query_images)
                  
                  prototypes = self.model.compute_prototypes(support_images, support_labels)
                  distances = euclidean_distance(query_embeddings, prototypes)
                  
                  predictions = distances.argmin(dim=1)
                elif self.config.META_LEARNING_ALGO == 'maml':
                  predictions = self.model.test(support_images, support_labels, query_images)
                elif self.config.META_LEARNING_ALGO == 'reptile':
                   predictions = self.model.test(support_images, support_labels, query_images)
                else:
                    raise ValueError(f"Meta learning algorithm '{self.config.META_LEARNING_ALGO}' not supported")
                accuracy = (predictions == query_labels).float().mean().item()
                
                accuracies.append(accuracy)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(query_labels.cpu().numpy())
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        confidence_interval = stats.t.interval(0.95, len(accuracies) - 1, loc=mean_accuracy, scale=std_accuracy/np.sqrt(len(accuracies)))
        
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        report = classification_report(all_labels, all_predictions, zero_division=0)
        print("\nClassification Report:")
        print(report)
        if self.config.USE_WANDB:
          wandb.log({
              "test_accuracy": mean_accuracy,
              "test_std": std_accuracy,
              "classification_report": report
          })
        
        return {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'confidence_interval': confidence_interval,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, true_labels: list, pred_labels: list) -> None:
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        
        cm_path = os.path.join(self.config.LOGS_DIR, f'confusion_matrix_test_{self.config.META_LEARNING_ALGO}.png')
        plt.savefig(cm_path)
        plt.close()

def main():
    config = Config()
    
    available_models = ['protonet', 'maml', 'reptile', 'best']
    model_type = input(f"Enter model type to test ({', '.join(available_models)}): ").lower()

    while model_type not in available_models:
        model_type = input(f"Invalid model type. Please select from ({', '.join(available_models)}): ").lower()

    tester = Tester(config, model_type)
    metrics = tester.test()
    print(f"\nTest Results:")
    print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    print(f"Standard Deviation: {metrics['std_accuracy']:.4f}")
    print(f"95% Confidence Interval: {metrics['confidence_interval']}")
    
    metrics_path = os.path.join(config.LOGS_DIR, f'metrics_test_{config.META_LEARNING_ALGO}.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)

if __name__ == '__main__':
    main()