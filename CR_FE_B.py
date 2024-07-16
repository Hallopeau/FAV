import torch
import torch.nn as nn
from tqdm import tqdm
import h5py
import pickle

from UCroma import PretrainedCROMA

class FExtractor:
    def __init__(self, dataloader, use_8_bit=True):
        self.dataloader = dataloader
        self.use_8_bit = use_8_bit
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FE = PretrainedCROMA(pretrained_path='CR.pt', size='base', modality='both', image_resolution=120)
        self.FE.to(self.device)
        self.FE.eval()

    def normalize(self, x):
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
        features_batches = []
        labels_batches = []
        id_batches = []
        with torch.no_grad():
            for optical_images, sar_images, labels, optical_img_paths, _ in tqdm(self.dataloader, desc="Extracting Features"):
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
                ids_i = torch.tensor([int((path.split(".")[0]).split("/")[-1]) for path in optical_img_paths])
                id_batches.append(ids_i)

        features = torch.cat(features_batches).numpy()
        labels = torch.cat(labels_batches).numpy()
        ids = torch.cat(id_batches).numpy()

        if save_name:
            hf = h5py.File(f'{save_name}.h5', 'w')
            hf.create_dataset('features', data=features)
            hf.create_dataset('labels', data=labels)
            hf.create_dataset('ids', data=ids)
            hf.close()

            with open(f'{save_name}.pkl', 'wb') as f:
                pickle.dump([features, labels, ids], f)

        return features, labels, ids
