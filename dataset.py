import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RandomDataGenerator(Dataset):
    def __init__(self, num_samples, image_size):
        self.num_samples = num_samples
        self.image_size = image_size
        self.landmarks_num = 19

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image
        random_image = np.random.randn(self.image_size, self.image_size)
        random_image = torch.tensor(random_image, dtype=torch.float32).unsqueeze(0).float()

        # Generate a random heatmap 
        heatmap =  torch.zeros(self.landmarks_num, self.image_size, self.image_size).unsqueeze(0).float()

        return random_image, heatmap

