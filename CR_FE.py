import torch
import torch.nn as nn
from tqdm import tqdm
import h5py
import pickle

from UCroma import PretrainedCROMA

class FExtractor:
    """
    A class to extract features from a dataset using the pretrained CROMA model.

    Attributes:
        dataloader: DataLoader for the dataset.
        use_8_bit: Flag to determine if 8-bit normalization should be used.
        device: Device on which to run the model (CPU or GPU).
        FE: Pretrained CROMA feature extractor model.
    """

    def __init__(self, dataloader, use_8_bit=True):
        """
        Initializes the FExtractor with the given parameters.

        Args:
            dataloader: DataLoader for the dataset.
            use_8_bit: Flag to determine if 8-bit normalization should be used. Defaults to True.
        """
        self.dataloader = dataloader
        self.use_8_bit = use_8_bit
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FE = PretrainedCROMA(pretrained_path='CR.pt', size='base', modality='both', image_resolution=120)
        self.FE.to(self.device)
        self.FE.eval()

    def normalize(self, x):
        """
        Normalizes the input images.

        Args:
            x: Input tensor of images.

        Returns:
            Normalized tensor of images.
        """
        x = x.float()
        imgs = []
        for channel in range(x.shape[1]):
            min_value = x[:, channel, :, :].mean() - 2 * x[:, channel, :, :].std()
            max_value = x[:, channel, :, :].mean() + 2 * x[:, channel, :, :].std()
            if self.use_8_bit:
                img = (x[:, channel, :, :] - min_value) / (max_value - min_value) * 255.0
                img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)
                imgs.append(img)
            else:
                img = (x[:, channel, :, :] - min_value) / (max_value - min_value)
                img = torch.clip(img, 0, 1).unsqueeze(dim=1)
                imgs.append(img)
        return torch.cat(imgs, dim=1)

    def extract_features(self, save_name=None):
        """
        Extracts features from the dataset and optionally saves them to a file.

        Args:
            save_name: Optional; Name of the file to save the features and labels.

        Returns:
            Tuple containing features and labels.
        """
        features_batches = []
        labels_batches = []

        with torch.no_grad():
            for optical_images, sar_images, labels, _, _ in tqdm(self.dataloader, desc="Extracting Features"):
                optical_images = optical_images.to(self.device)
                optical_images = self.normalize(optical_images)
                sar_images = sar_images.to(self.device)
                sar_images = self.normalize(sar_images)
                if self.use_8_bit:
                    optical_images = optical_images.float() / 255
                    sar_images = sar_images.float() / 255
                outputs = self.FE(SAR_images=sar_images, optical_images=optical_images)['joint_GAP']
                features_batches.append(outputs.cpu())
                labels_batches.append(labels)

        features = torch.cat(features_batches).numpy()
        labels = torch.cat(labels_batches).numpy()

        if save_name:
            with h5py.File(f'{save_name}.h5', 'w') as hf:
                hf.create_dataset('features', data=features)
                hf.create_dataset('labels', data=labels)

            with open(f'{save_name}.pkl', 'wb') as f:
                pickle.dump([features, labels], f)

        return features, labels
