import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch

#================================================================#

class NormalDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing subdirectories of images, each named by their label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label_map = {'normal': 0, 'diseases': 1}

        for label in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, label + '/')
            if os.path.isdir(dir_path):
                for img_file in os.listdir(dir_path):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        self.samples.append((os.path.join(dir_path, img_file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        label = self.label_map.get(label, -1)
        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        
        return image, label
    