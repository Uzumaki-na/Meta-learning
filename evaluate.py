import torch
from config import Config
from data.omniglot_download import OmniglotDownloader
from utils.data_loader import OmniglotDataset, create_episode, get_default_transform
from models.protonet import PrototypicalNetwork
from models.maml import MAML
from models.reptile import Reptile
from utils.training_utils import euclidean_distance
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import yaml
from typing import Dict, List

def evaluate(model, dataset, n_way, k_shot, q_query, meta_algo, num_episodes=100):
    model.eval()
    all_predictions = []
    all_labels = []
    accuracies = []
    
    with torch.no_grad():
        for _ in range(num_episodes):
            support_images, support_labels, query_images, query_labels = create_episode(dataset, n_way, k_shot, q_query)
            support_images, support_labels = support_images.to(Config().DEVICE), support_labels.to(Config().DEVICE)
            query_images, query_labels = query_images.to(Config().DEVICE), query_labels.to(Config().DEVICE)
            
            if meta_algo == 'protonet':
              query_embeddings = model(query_images)
              prototypes = model.compute_prototypes(support_images, support_labels)
              distances = euclidean_distance(query_embeddings, prototypes)
              predictions = distances.argmin(dim=1)
            elif meta_algo == 'maml':
              predictions = model.test(support_images, support_labels, query_images)
            elif meta_algo == 'reptile':
               predictions = model.test(support_images, support_labels, query_images)
            else:
                raise ValueError(f"Meta learning algorithm '{meta_algo}' not supported")
            
            accuracy = (predictions == query_labels).float().mean().item()
            accuracies.append(accuracy)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    confidence_interval = stats.t.interval(0.95, len(accuracies) - 1, loc=mean_accuracy, scale=std_accuracy/np.sqrt(len(accuracies)))
    
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    print(f'Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    print(f"95% Confidence Interval: {confidence_interval}")
    
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    config = Config()
    cm_path = os.path.join(config.LOGS_DIR, f'confusion_matrix_evaluate_{meta_algo}.png')
    plt.savefig(cm_path)
    plt.close()
    
    return {
      'accuracy': mean_accuracy,
      'accuracy_std': std_accuracy,
      'f1': f1,
      'precision': precision,
      'recall': recall,
      'confidence_interval': confidence_interval,
    }

def main():
    config = Config()
    
    available_models = ['protonet', 'maml', 'reptile']
    model_type = input(f"Enter model type to evaluate ({', '.join(available_models)}): ").lower()
    
    while model_type not in available_models:
        model_type = input(f"Invalid model type. Please select from ({', '.join(available_models)}): ").lower()
    
    config.META_LEARNING_ALGO = model_type
    dataset_path = OmniglotDownloader.download_dataset(config.DATASET_PATH)
    transform = get_default_transform(config)
    dataset = OmniglotDataset(root_dir=dataset_path, transform=transform)
    
    if config.META_LEARNING_ALGO == 'protonet':
      model = PrototypicalNetwork(config)
    elif config.META_LEARNING_ALGO == 'maml':
      model = MAML(config)
    elif config.META_LEARNING_ALGO == 'reptile':
        model = Reptile(config)
    else:
      raise ValueError(f"Meta learning algorithm '{config.META_LEARNING_ALGO}' not supported")
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_{config.META_LEARNING_ALGO}.pth')
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])  
    model.to(config.DEVICE)
    
    metrics = evaluate(model, dataset, config.N_WAY, config.K_SHOT, config.Q_QUERY, config.META_LEARNING_ALGO)
    print(f'Accuracy: {metrics["accuracy"]:.4f}')

    metrics_path = os.path.join(config.LOGS_DIR, f'metrics_evaluate_{config.META_LEARNING_ALGO}.yaml')
    with open(metrics_path, 'w') as f:
      yaml.dump(metrics, f, default_flow_style=False)

if __name__ == '__main__':
    main()