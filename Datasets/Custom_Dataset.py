import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    '''
    Custom dataset class for the lazy loading of the data

    - type: Generated or Real Life

    - set: train, validation or test

    - balance: If true, applies oversampling AND data augmentation to the minority classes

    - filter_array: If not empty, only the images with the indexes in the array will be retained
    '''
    def __init__(self, type, set, balance=True, filter_array = []):

        # Check data validity
        if type not in ["Generated", "Real Life"]:
            raise ValueError("Dataset not valid")
        if set not in ["train", "validation", "test"]:
            raise ValueError("Set not valid")

        self.type = type
        self.set = set
        self.balance = balance

        if self.balance:
            # Transform to apply to the minibatches for data augmentation
            # Define the transformation to apply
            # Transformations: Random horizontal and vertical flips, halving and doubling the brightness
            # This should improve the prediction accuracy
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((100, 100), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=[0.75, 1.25])], p=0.5)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((100, 100), antialias=True)
            ])
            
        # Define the data_path depending on the type of the dataset and the set
        # Note the data path is relative to the file that is calling the dataset class
        # The calls are made in MODEL/MODEL_NAME/Model.py
        if self.type == "Generated":
            self.root_path = "../../Datasets/Generated Data/"
            if self.set == "train":
                self.data_path = "../../Datasets/Generated Data/train_full_generated_data.json"
            elif self.set == "validation":
                self.data_path = "../../Datasets/Generated Data/validation_full_generated_data.json"
            else:
                self.data_path = "../../Datasets/Generated Data/test_full_generated_data.json"
        else:
            self.root_path = "../../Datasets/Real Life Data/"
            if self.set == "train":
                self.data_path = "../../Datasets/Real Life Data/train_full_real_life_data.json"
            elif self.set == "validation":
                self.data_path = "../../Datasets/Real Life Data/validation_full_real_life_data.json"
            else:
                self.data_path = "../../Datasets/Real Life Data/test_full_real_life_data.json"
        
        # Load the JSON file
        with open(self.data_path, "r") as file:
            self.data = json.load(file)
        
        # Get the images name form the JSON file 
        self.images = np.array(self.data[self.set])
        
        # Get the labels from the JSON file
        self.labels = self.data["label"]
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # If the retain array is not empty, only retain the images with the indexes in the array
        if len(filter_array) > 0:
            self.images = self.images[filter_array]
            self.labels = self.labels[filter_array]
        
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
def get_loader(dataset, batch_size, balance = True):

    # # Because we are using balanced accuracy scores, we can use the class analytics gathered during pre-processing to define the following class distribution array:
    # class_proportions_gen = np.array([0.3198, 0.1602, 0.0405, 0.0400, 0.0406, 0.0201, 0.0404, 0.1596, 0.0392, 0.0397, 0.0400, 0.0196, 0.0404])
    if(dataset.type == "Generated"):
        
        # Load the class distribution from the JSON file
        with open("../../Datasets PreProcessing/Data Generation/class_distribution.json", "r") as file:
            class_proportions = np.array(list(json.load(file)[dataset.set].values()))
        
        # Compute the percentages
        class_proportions = class_proportions / np.sum(class_proportions)
        
    else:
        # Load the class distribution from the JSON file
        with open("../../Datasets PreProcessing/Real life data/class_distribution.json", "r") as file:
            class_proportions = np.array(list(json.load(file)[dataset.set].values()))

        # Compute the percentages
        class_proportions = class_proportions / np.sum(class_proportions)

    # Define the sampler using class distributions to oversample the minority classes
    if balance:
        class_weights = 1. / torch.tensor(class_proportions, dtype=torch.float) # The weights of the classes
        sample_weights = class_weights[dataset.labels] # Assign each label its corresponding weight
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(dataset), replacement=True)
    else:
        sampler = None
        
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)