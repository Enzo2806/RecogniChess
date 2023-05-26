import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class CustomDataset(Dataset):
    '''
    Custom dataset class for the lazy loading of the data

    - dataset: Generated or Real Life

    - set: train, validation or test

    - full_dataset: True if the full dataset is used, False if the balanced dataset is used

    - transform: True if the data augmentation is applied, False otherwise
    '''
    def __init__(self, dataset, set, full_dataset = False, apply_transform = True):

        # Check data validity
        if dataset not in ["Generated", "Real Life"]:
            raise ValueError("Dataset not valid")
        if set not in ["train", "validation", "test"]:
            raise ValueError("Set not valid")

        self.dataset = dataset
        self.set = set
        self.full_dataset = full_dataset
        self.apply_transform = apply_transform

        if self.apply_transform:
            # Transform to apply to the minibatches for data augmentation
            # Define the transformation to apply
            # Transformations: Random horizontal and vertical flips, halving and doubling the brightness
            # This should improve the prediction accuracy
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((100, 100)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=[0.75, 1.25])], p=0.5)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
            
        # Define the data_path depending on the dataset, the set and the full_dataset flag
        # Note the data path is relative to the file that is calling the dataset class
        # The calls are made in MODEL/MODEL_NAME/Model.py
        if self.dataset == "Generated":
            self.root_path = "../../Datasets/Generated Data/"
            if self.set == "train":
                self.data_path = "../../Datasets/Generated Data/train_full_generated_data.json" if self.full_dataset else \
                                 "../../Datasets/Generated Data/train_balanced_generated_data.json"
            elif self.set == "validation":
                self.data_path = "../../Datasets/Generated Data/validation_full_generated_data.json" if self.full_dataset else \
                                "../../Datasets/Generated Data/validation_balanced_generated_data.json"
            else:
                self.data_path = "../../Datasets/Generated Data/test_full_generated_data.json" if self.full_dataset else \
                                "../../Datasets/Generated Data/test_balanced_generated_data.json"
        else:
            self.root_path = "../../Datasets/Real Life Data/"
            if self.set == "train":
                self.data_path = "../../Datasets/Real Life Data/train_full_real_life_data.json" if self.full_dataset else \
                                "../../Datasets/Real Life Data/train_balanced_real_life_data.json"
            elif self.set == "validation":
                self.data_path = "../../Datasets/Real Life Data/validation_full_real_life_data.json" if self.full_dataset else \
                                "../../Datasets/Real Life Data/validation_balanced_real_life_data.json"
            else:
                self.data_path = "../../Datasets/Real Life Data/test_full_real_life_data.json" if self.full_dataset else \
                                "../../Datasets/Real Life Data/test_balanced_real_life_data.json"
        
        # Load the JSON file
        with open(self.data_path, "r") as file:
            self.data = json.load(file)
        
        # Get the images name form the JSON file 
        self.images = self.data[self.set]
        
        # Get the labels from the JSON file
        self.labels = self.data["label"]

        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.images[idx])

        image = Image.open(img_path)
        image = self.transform(image)
        
        # Convert the label to int instead of float
        label = self.labels[idx]
        
        return image, label

# We can now proceed to defining a function that creates a data loader for both datasets, oversampling the minority classes and applying horizontal flip and blur transformations:
def get_gen_loader(dataset, batch_size):

    # Because we are using balanced accuracy scores, we can use the class analytics gathered during pre-processing to define the following class distribution array:
    class_proportions_gen = np.array([0.3198, 0.1602, 0.0405, 0.0400, 0.0406, 0.0201, 0.0404, 0.1596, 0.0392, 0.0397, 0.0400, 0.0196, 0.0404])
    class_proportions_real = np.array([0.3228, 0.1738, 0.0347, 0.0415, 0.0454, 0.0206, 0.0354, 0.1490, 0.0284, 0.0463, 0.0432, 0.0234, 0.0354])

    # Define the sampler using class distributions to oversample the minority classes
    class_weights = 1. / torch.tensor(class_proportions_gen, dtype=torch.float) # The weights of the classes
    sample_weights = class_weights[dataset.labels] # Assign each label its corresponding weight
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def get_real_loader(dataset, batch_size):

    # Because we are using balanced accuracy scores, we can use the class analytics gathered during pre-processing to define the following class distribution array:
    class_proportions_gen = np.array([0.3198, 0.1602, 0.0405, 0.0400, 0.0406, 0.0201, 0.0404, 0.1596, 0.0392, 0.0397, 0.0400, 0.0196, 0.0404])
    class_proportions_real = np.array([0.3228, 0.1738, 0.0347, 0.0415, 0.0454, 0.0206, 0.0354, 0.1490, 0.0284, 0.0463, 0.0432, 0.0234, 0.0354])

    # Define the sampler using class distributions to oversample the minority classes
    class_weights = 1. / torch.tensor(class_proportions_real, dtype=torch.float) # The weights of the classes
    sample_weights = class_weights[dataset.labels] # Assign each label its corresponding weight
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)