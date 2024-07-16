import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from PIL import Image
import imageio.v2 as imageio

class Loader:
    """
    A class to load optical and SAR image data, apply transformations, and create data loaders.

    Attributes:
        opt_root_dir: Root directory containing optical images.
        sar_root_dir: Root directory containing SAR images.
        num_folds: Number of folds for cross-validation.
        loaders: List of DataLoader objects for each fold.
        sar_loaders: List of DataLoader objects for SAR images (not used in this code).
    """

    def __init__(self, opt_root_dir, sar_root_dir, num_folds=5):
        """
        Initializes the Loader with the given parameters.

        Args:
            opt_root_dir: Root directory containing optical images.
            sar_root_dir: Root directory containing SAR images.
            num_folds: Number of folds for cross-validation.
        """
        self.opt_root_dir = opt_root_dir
        self.sar_root_dir = sar_root_dir
        self.num_folds = num_folds
        self.loaders = []
        self.sar_loaders = []

    def load_data(self, sample_size=224, batch_size=32, data_seed=None, split_seed=None):
        """
        Loads data, applies transformations, and creates data loaders.

        Args:
            sample_size: Size to which each image will be resized.
            batch_size: Number of samples per batch.
            data_seed: Seed for random sampling of images.
            split_seed: Seed for splitting the dataset into folds.

        Returns:
            List of DataLoader objects for each fold.
        """
        class CustomDataset(VisionDataset):
            """
            Custom dataset class for loading optical and SAR images.
            """

            def __init__(self, opt_root, sar_root, transform=None, target_transform=None):
                """
                Initializes the CustomDataset with the given parameters.

                Args:
                    opt_root: Root directory containing optical images.
                    sar_root: Root directory containing SAR images.
                    transform: Transformations to be applied to the images.
                    target_transform: Transformations to be applied to the targets.
                """
                super(CustomDataset, self).__init__(root=opt_root, transform=transform, target_transform=target_transform)
                self.opt_root = opt_root
                self.sar_root = sar_root
                self.transform = transform
                self.target_transform = target_transform

                self.opt_classes = sorted(os.listdir(opt_root))
                self.sar_classes = sorted(os.listdir(sar_root))

                assert self.opt_classes == self.sar_classes, "Optical and SAR classes do not match."
                self.classes = self.opt_classes
                
                min_num_images = float('inf')
                for class_name in self.classes:
                    class_path = os.path.join(opt_root, class_name)
                    if os.path.isdir(class_path):
                        num_images = len(os.listdir(class_path))
                        min_num_images = min(min_num_images, num_images)

                self.samples = []
                for class_name in self.opt_classes:
                    opt_class_path = os.path.join(opt_root, class_name)
                    sar_class_path = os.path.join(sar_root, class_name)

                    if os.path.isdir(opt_class_path) and os.path.isdir(sar_class_path):
                        opt_images = [img_name for img_name in os.listdir(opt_class_path) if os.path.isfile(os.path.join(opt_class_path, img_name))]
                        sar_images = [img_name for img_name in os.listdir(sar_class_path) if os.path.isfile(os.path.join(sar_class_path, img_name))]
                        
                        if data_seed is not None:
                            random.seed(data_seed)
                        selected_images = random.sample(opt_images, min_num_images)
                        for img_name in selected_images:
                            opt_img_path = os.path.join(opt_class_path, img_name)
                            sar_img_path = os.path.join(sar_class_path, img_name)
                            self.samples.append((opt_img_path, sar_img_path, self.opt_classes.index(class_name)))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                opt_img_path, sar_img_path, target = self.samples[idx]

                opt_img = imageio.imread(opt_img_path)
                sar_img = imageio.imread(sar_img_path)

                opt_img = np.transpose(opt_img, (2, 0, 1)).astype(np.float32)
                sar_img = np.transpose(sar_img, (2, 0, 1)).astype(np.float32)

                opt_img = torch.tensor(opt_img)
                sar_img = torch.tensor(sar_img)

                if self.transform:
                    opt_img = self.transform(opt_img)
                    sar_img = self.transform(sar_img)

                if self.target_transform:
                    target = self.target_transform(target)

                return opt_img, sar_img, target, opt_img_path, sar_img_path

        transform = transforms.Resize((sample_size, sample_size), antialias=True)
        combined_dataset = CustomDataset(opt_root=self.opt_root_dir, sar_root=self.sar_root_dir, transform=transform)
        self.combined_dataset = combined_dataset

        fold_sizes = [len(combined_dataset) // self.num_folds] * self.num_folds
        remainder = len(combined_dataset) % self.num_folds
        for i in range(remainder):
            fold_sizes[i] += 1

        if split_seed is not None:
            torch.manual_seed(split_seed)
        folds = random_split(combined_dataset, fold_sizes)

        loaders = []
        for fold in folds:
            loaders.append(DataLoader(fold, batch_size=batch_size, shuffle=True))
        self.loaders = loaders
        