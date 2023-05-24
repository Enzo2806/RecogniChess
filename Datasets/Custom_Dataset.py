import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Transform to apply to the minibatches for data augmentation
# Define the transformation to apply
# Transformations: Random horizontal and vertical flips, halving and doubling the brightness
# This should improve the prediction accuracy
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=[0.75, 1.25])], p=0.5)
])

class CustomDataset(Dataset):
    '''
    Custom dataset class for the lazy loading of the data

    '''
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

        transform = transforms.ToTensor()
        image = transform(image)
        
        label = self.labels[idx]
        
        return image, label
    
train_dataset = CustomLazyDataset("train_full_generated_data.json", "../Datasets/Generated Data/")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

class CustomDataset(Dataset):
    
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.images = os.listdir(root_dir)
        self.labels = torch.load(label_dir).long()

        # Remove the labels that do not belong to this split of the dataset (Labels is all labels)
        self.labels = self.labels[torch.tensor([int(img_name[3:9]) for img_name in self.images])]



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        image = transform(image)
        label = self.labels[idx] # Since we removed the labels that do not belong to this split, we can use idx directly
        return image, label

# Extract the generated data
generated_data_root = "../../Data Generation/Pre Processed Data Generated"
train_gen_dataset = CustomDataset(generated_data_root + "/Square Images/Training", generated_data_root + "/Square Images/y_generated.pt")
val_gen_dataset = CustomDataset(generated_data_root + "/Square Images/Validation", generated_data_root + "/Square Images/y_generated.pt")
test_gen_dataset = CustomDataset(generated_data_root + "/Square Images/Testing", generated_data_root + "/Square Images/y_generated.pt")

# Extract the real data
real_data_root = "../../Real life data/Pre processed Real Life"
train_real_dataset = CustomDataset(real_data_root + "/Square Images/Training", real_data_root + "/Square Images/y_real_life.pt")
val_real_dataset = CustomDataset(real_data_root + "/Square Images/Validation", real_data_root + "/Square Images/y_real_life.pt")
test_real_dataset = CustomDataset(real_data_root + "/Square Images/Testing", real_data_root + "/Square Images/y_real_life.pt")