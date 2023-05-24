import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomLazyDataset(Dataset):
    def __init__(self, data_file, root_dir):
        self.data_file = data_file
        self.root_dir = root_dir
        
        # Load the JSON file
        with open(self.data_file, "r") as file:
            self.data = json.load(file)
        
        self.images = self.data["train"] if self.data_file == "train_full_generated_data.json" else \
                      self.data["validation"] if self.root_dir == "validation_full_generated_data.json" else \
                      self.data["test"]
        
        self.labels = self.data["label"]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        # Perform any necessary image transformations here
        transform = transforms.ToTensor()
        image = transform(image)
        
        label = self.labels[idx]
        
        return image, label
    
train_dataset = CustomLazyDataset("train_full_generated_data.json", "../Datasets/Generated Data/")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)