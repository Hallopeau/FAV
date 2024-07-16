import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import os
import random

class Loader:
    def __init__(self, root_dir, num_folds=5):
        """
        Initializes the Loader class.

        Parameters:
        root_dir (str): Directory with all the images, organized by class.
        num_folds (int): Number of folds to split the data into.
        """
        self.root_dir = root_dir
        self.num_folds = num_folds
        self.loaders = []

    def load_data(self, sample_size=224, batch_size=32, data_seed=None, split_seed=None):
        """
        Loads the data, applies transformations, and splits it into folds.

        Parameters:
        sample_size (int): Size to resize the images to (sample_size x sample_size).
        batch_size (int): Number of images per batch.
        data_seed (int, optional): Seed for random sampling of images.
        split_seed (int, optional): Seed for random splitting into folds.
        """
        class CustomDataset(VisionDataset):
            def __init__(self, root, transform=None, target_transform=None):
                """
                Custom dataset for loading images.

                Parameters:
                root (str): Root directory of the dataset.
                transform (callable, optional): Optional transform to be applied on a sample.
                target_transform (callable, optional): Optional transform to be applied on a target.
                """
                super(CustomDataset, self).__init__(root, transform=transform, target_transform=target_transform)
                self.root = root
                self.transform = transform
                self.target_transform = target_transform
                self.classes = sorted(os.listdir(root))
                
                min_num_images = float('inf')
                
                for class_name in self.classes:
                    class_path = os.path.join(root, class_name)
                    if os.path.isdir(class_path):
                        num_images = len(os.listdir(class_path))
                        min_num_images = min(min_num_images, num_images)

                self.samples = []
                        
                for class_name in self.classes:
                    class_path = os.path.join(root, class_name)
                    if os.path.isdir(class_path):
                        images = [img_name for img_name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img_name))]
                        if data_seed is not None:
                            random.seed(data_seed)
                        selected_images = random.sample(images, min_num_images)
                        for img_name in selected_images:
                            img_path = os.path.join(class_path, img_name)
                            self.samples.append((img_path, self.classes.index(class_name)))

            def __len__(self):
                """
                Returns the total number of samples.
                """
                return len(self.samples)

            def __getitem__(self, idx):
                """
                Fetches the sample and target at the given index.

                Parameters:
                idx (int): Index of the sample to fetch.

                Returns:
                tuple: (image, target, image_path)
                """
                img_path, target = self.samples[idx]
                img = imageio.imread(img_path)[:, :, :3]
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                if self.transform:
                    img = self.transform(img)
                if self.target_transform:
                    target = self.target_transform(target)
                return img, target, img_path

        transform = transforms.Compose([
            transforms.Resize((sample_size, sample_size)),
            transforms.ToTensor()
        ])

        dataset = CustomDataset(root=self.root_dir, transform=transform)
        self.dataset = dataset

        fold_sizes = [len(dataset) // self.num_folds] * self.num_folds
        remainder = len(dataset) % self.num_folds
        for i in range(remainder):
            fold_sizes[i] += 1

        if split_seed is not None:
            torch.manual_seed(split_seed)
        folds = random_split(dataset, fold_sizes)

        loaders = []
        for f in folds: 
            loaders.append(DataLoader(f, batch_size=batch_size, shuffle=True))
        self.loaders = loaders
