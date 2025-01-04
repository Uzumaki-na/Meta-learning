import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from config import Config  

class OmniglotDataset(Dataset):
    def __init__(self, root_dir, split='background', transform=None, cache_images=True, preload_images=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes = self._get_classes()
        self.cache_images = cache_images
        self.preload_images = preload_images
        self.image_cache = {}
        self.preloaded_images = {}

        if preload_images:
            self._preload_all_images()

    def _get_classes(self):
        classes = []
        dataset_path = self.root_dir
        for alphabet in os.listdir(dataset_path):
            alphabet_path = os.path.join(dataset_path, alphabet)
            if os.path.isdir(alphabet_path):
                for character in os.listdir(alphabet_path):
                    character_path = os.path.join(alphabet_path, character)
                    if os.path.isdir(character_path):
                        classes.append(character_path)
        if not classes:
            raise ValueError(f"No classes found in the dataset path: {dataset_path}")
        return classes

    def _preload_all_images(self):
      print("Preloading images in RAM. This may take some time.")
      with ProcessPoolExecutor() as executor:
        futures = {
          executor.submit(self._load_image_and_transform, class_idx): class_idx for class_idx in range(len(self.classes))
        }
        
        preloaded_images = {} 
        for future in tqdm(as_completed(futures), total=len(self.classes), desc="Preloading"):
          class_idx = futures[future]
          try:
             preloaded_images[class_idx] = future.result()
          except Exception as e:
              print(f"Error preloading {self.classes[class_idx]}: {e}")
        
        self.preloaded_images = preloaded_images 

    def _load_image_and_transform(self, idx):
        class_path = self.classes[idx]
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
          raise ValueError(f"No images found in class path: {class_path}")
        all_images = []
        for img_path in images:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            all_images.append(image)
        return all_images
    
    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        if self.preload_images:
            try:
              images = self.preloaded_images[idx]
              img = random.choice(images)
            except Exception as e:
              print(f'Error loading from preloaded dataset: {e}. Reverting to normal load')
              class_path = self.classes[idx]
              images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
              if not images:
                  raise ValueError(f"No images found in class path: {class_path}")
              img_path = np.random.choice(images)
              image = Image.open(img_path).convert('L')
              if self.transform:
                img = self.transform(image)
        else:
            class_path = self.classes[idx]
            if self.cache_images and class_path in self.image_cache:
                images = self.image_cache[class_path]
                img = random.choice(images)
            else:
                images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not images:
                    raise ValueError(f"No images found in class path: {class_path}")
                img_path = np.random.choice(images)
                image = Image.open(img_path).convert('L')
                if self.transform:
                  img = self.transform(image)
                if self.cache_images:
                    self.image_cache[class_path] = [img] 
        return img, idx

def get_default_transform(config: Config):
    if config.USE_DATA_AUGMENTATION:
      if config.USE_DATA_AUGMENTATION == 'rand':  
        return transforms.Compose([
          transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
          transforms.RandomRotation(degrees=15),
          transforms.RandomResizedCrop(size=(config.INPUT_SIZE, config.INPUT_SIZE), scale=(0.8, 1.0)),
          rand_augment_transform('rand-m9-mstd0.5', 
                                    hparams={'translate_const': config.INPUT_SIZE // 4, 'img_mean': (127,)}),
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,)),
        ])
      elif config.USE_DATA_AUGMENTATION == 'trivial':  
          return transforms.Compose([
          transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
          transforms.RandomRotation(degrees=15),
          transforms.RandomResizedCrop(size=(config.INPUT_SIZE, config.INPUT_SIZE), scale=(0.8, 1.0)),
          auto_augment_transform('trivialaugment',
                                 hparams={'translate_const': config.INPUT_SIZE // 4, 'img_mean': (127,)}),
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,)),
        ])
      else:
        return transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=(config.INPUT_SIZE, config.INPUT_SIZE), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

def create_episode(dataset: OmniglotDataset, n_way: int, k_shot: int, q_query: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_classes = random.sample(range(len(dataset)), n_way)
    
    support_images = []
    query_images = []
    support_labels = []
    query_labels = []

    for i, class_idx in enumerate(selected_classes):
        try:
           if dataset.preload_images:
              all_images = dataset.preloaded_images[class_idx]
           else:
               class_path = dataset.classes[class_idx]
               images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
               all_images = []
               for img_path in images:
                    img = Image.open(img_path).convert('L')
                    if dataset.transform:
                       img = dataset.transform(img)
                    all_images.append(img)
        except Exception as e:
            print(f"Error loading from preloaded, reverting to normal: {e}")
            class_path = dataset.classes[class_idx]
            images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            if not images:
              raise ValueError(f"No images found in class path: {class_path}")
            all_images = []
            for img_path in images:
                img = Image.open(img_path).convert('L')
                if dataset.transform:
                    img = dataset.transform(img)
                all_images.append(img)
        
        all_images = np.array(all_images)
        
        num_images = len(all_images)
        selected_indices = np.random.choice(num_images, k_shot + q_query, replace=False) 
        support_images.extend(all_images[selected_indices[:k_shot]])
        query_images.extend(all_images[selected_indices[k_shot:]])
        
        support_labels.extend([i] * k_shot)
        query_labels.extend([i] * q_query)
    
    return torch.stack([torch.from_numpy(img) for img in support_images]), torch.tensor(support_labels, dtype=torch.long), torch.stack([torch.from_numpy(img) for img in query_images]), torch.tensor(query_labels, dtype=torch.long)

def _collate_fn(batch):
    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for item in batch:
        support, support_label, query, query_label = item 
        support_images.append(support)
        support_labels.append(support_label)
        query_images.append(query)
        query_labels.append(query_label)

    return (torch.cat(support_images), torch.cat(support_labels), 
            torch.cat(query_images), torch.cat(query_labels))


def create_data_loader(dataset: OmniglotDataset, n_way: int, k_shot: int, q_query: int, batch_size: int, num_workers: int = 4):
    
    episodes = []
    for _ in range(10000):
        episodes.append(create_episode(dataset, n_way, k_shot, q_query)) 
    
    return DataLoader(
        episodes, 
        batch_size=batch_size, 
        collate_fn=_collate_fn,  
        num_workers=num_workers,
        shuffle=True
    )