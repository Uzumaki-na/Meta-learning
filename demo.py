import streamlit as st
import torch
from PIL import Image
import numpy as np
from typing import List
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import os
import yaml

from config import Config
from models.protonet import PrototypicalNetwork
from models.maml import MAML
from models.reptile import Reptile
from data.omniglot_download import OmniglotDownloader
from utils.data_loader import OmniglotDataset, create_episode, get_default_transform
from utils.training_utils import euclidean_distance


def load_model(config, model_type):
    """Load the trained model."""
    if model_type == 'protonet':
      model = PrototypicalNetwork(config)
    elif model_type == 'maml':
      model = MAML(config)
    elif model_type == 'reptile':
      model = Reptile(config)
    else:
        raise ValueError(f"Meta learning algorithm '{model_type}' not supported")
        
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_{model_type}.pth')
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    return model

def preprocess_image(image, config):
    """Preprocess the input image."""
    transform = get_default_transform(config)
    image = image.convert('L')  # Convert to grayscale
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension


def get_prediction(model, support_images, support_labels, query_image, config, meta_learning_algo):
    """Get the prediction for the query image."""
    with torch.no_grad():
      query_image = query_image.to(config.DEVICE)
      support_images = support_images.to(config.DEVICE)
      support_labels = support_labels.to(config.DEVICE)

      if meta_learning_algo == 'protonet':
        query_embedding = model(query_image)
        prototypes = model.compute_prototypes(support_images, support_labels)
        distances = euclidean_distance(query_embedding, prototypes)
        prediction = distances.argmin(dim=1)
      elif meta_learning_algo == 'maml':
          prediction = model.test(support_images, support_labels, query_image)
      elif meta_learning_algo == 'reptile':
           prediction = model.test(support_images, support_labels, query_image)
      else:
         raise ValueError(f"Meta learning algorithm '{meta_learning_algo}' not supported")
      
    return prediction.item()

def visualize_embeddings(model, images, labels, config):
    """Visualizes the embeddings using PCA."""
    with torch.no_grad():
      embeddings = []
      for img in images:
         img = img.unsqueeze(0).to(config.DEVICE)
         embedding = model(img).cpu().numpy()
         embeddings.append(embedding)
      embeddings = np.concatenate(embeddings, axis=0)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Visualization of Learned Embeddings")
    st.pyplot(plt.gcf(), clear_figure=True)

def main():
    st.title("Omniglot Few-Shot Character Recognition Demo")
    config = Config()

    #Model selection
    available_models = ['protonet', 'maml', 'reptile', 'best']
    model_type = st.selectbox(f"Select a model to use", available_models)
    
    if model_type == 'best':
      best_model_metrics = {}
      best_model_type = None
      for model_type in ['protonet', 'maml', 'reptile']:
        try:
          metrics_path = os.path.join(config.LOGS_DIR, f'metrics_evaluate_{model_type}.yaml')
          with open(metrics_path, 'r') as f:
               metrics = yaml.safe_load(f)
          best_model_metrics[model_type] = metrics['accuracy']
        except FileNotFoundError:
           best_model_metrics[model_type] = -1
      best_model_type = max(best_model_metrics, key = best_model_metrics.get)
      if best_model_metrics[best_model_type] == -1:
         raise ValueError("No models available for evaluation")
      config.META_LEARNING_ALGO = best_model_type
    else:
        config.META_LEARNING_ALGO = model_type
    #Load dataset and model
    dataset_path = OmniglotDownloader.download_dataset(config.DATASET_PATH)
    transform = get_default_transform(config)
    dataset = OmniglotDataset(root_dir=dataset_path, transform=transform, preload_images=True)
    model = load_model(config, config.META_LEARNING_ALGO)
    
    #Sidebar
    st.sidebar.header("Settings")
    n_way = st.sidebar.slider("N-way", min_value=2, max_value=10, value=5)
    k_shot = st.sidebar.slider("K-shot", min_value=1, max_value=5, value=1)
    
    #Create an episode
    support_images, support_labels, _, _ = create_episode(dataset, n_way, k_shot, 1)
    
    # Display support images
    st.subheader("Support Images")
    cols = st.columns(len(support_images))
    for i, img in enumerate(support_images):
        img_pil = Image.fromarray((img.cpu().numpy() * 127.5 + 127.5).astype(np.uint8).squeeze()) #De-normalize the image
        cols[i].image(img_pil, caption=f"Class {support_labels[i]}")
    
    # Get query image from user
    uploaded_file = st.file_uploader("Upload a Query Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image, config)
        
        prediction = get_prediction(model, support_images, support_labels, processed_image, config, config.META_LEARNING_ALGO)
        st.subheader("Prediction")
        st.write(f"The image is predicted to belong to class: {prediction}")
    
    if st.checkbox("Show Embeddings"):
      st.subheader("Embedding Visualization")
      all_images = []
      all_labels = []
      num_classes = len(dataset.classes)
      for i in range(num_classes):
          img, label = dataset[i]
          all_images.append(img)
          all_labels.append(label)
      visualize_embeddings(model, all_images, all_labels, config)

if __name__ == "__main__":
    main()