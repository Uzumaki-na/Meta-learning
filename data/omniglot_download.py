import os
import urllib.request
import zipfile
import torch
from torchvision import transforms

class OmniglotDownloader:
    @staticmethod
    def download_dataset(save_path, dataset_type='background'):
        os.makedirs(save_path, exist_ok=True)
        
        if dataset_type == 'background':
            url = "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip"
            file_name = "images_background.zip"
        elif dataset_type == 'evaluation':
            url = "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"
            file_name = "images_evaluation.zip"
        else:
            raise ValueError("Invalid dataset_type. Use 'background' or 'evaluation'.")
        
        file_path = os.path.join(save_path, file_name)
        extracted_path = os.path.join(save_path, dataset_type)
        
        if not os.path.exists(extracted_path):
            print(f"Downloading {dataset_type} dataset...")
            
            if not os.path.exists(file_path):
                urllib.request.urlretrieve(url, file_path)
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(save_path)
            
            original_extracted_path = os.path.join(save_path, 'images_' + dataset_type)
            if os.path.exists(original_extracted_path):
                os.rename(original_extracted_path, extracted_path)
        
        return extracted_path