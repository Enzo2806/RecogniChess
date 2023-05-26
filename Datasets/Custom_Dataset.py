import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

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
    
# Uncomment to test the custom dataset
# def try_dataset(dataset, set, full_dataset = False, apply_transform = True):
#     dataset = CustomDataset(dataset, set)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

#     # Get the first batch of data
#     images, labels = next(iter(loader))

#     # Plot the images with their labels
#     fig, axs = plt.subplots(4, 8, figsize=(12, 6))
#     fig.tight_layout()

#     for i in range(4):
#         for j in range(8):
#             index = i * 8 + j
#             image = images[index]
#             label = labels[index]

#             # Reverse any preprocessing or transformation applied to the image
#             # (if applicable) before plotting

#             axs[i][j].imshow(image.permute(1, 2, 0))
#             axs[i][j].set_title(f"Label: {label}")
#             axs[i][j].axis("off")

#     plt.show()

# try_dataset("Generated", "train", full_dataset = True, apply_transform = False)