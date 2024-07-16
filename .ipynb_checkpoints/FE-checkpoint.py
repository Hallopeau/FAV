import h5py
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import timm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class FeatureExtractor:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name not in timm.list_models(pretrained=True):
            print("Error, '%s' is not a valid model name for timm library. For a list of available pretrained models, "
                  "run: \n'timm.list_models(pretrained=True)'" % model_name)
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()

        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.model_transform = timm.data.create_transform(**data_cfg)

        self.file_name = None

    def get_features(self, dataloader, name=None):
        if name != None:
            self.file_name = "features_%s_%s.h5" % (name, self.model_name)
            if os.path.exists(self.file_name):
                features, labels = self._load_features(self.file_name)
            else:
                features, labels = self._extract_features(dataloader)
                self._save_features(features, labels, self.file_name)
        else:
            features, labels = self._extract_features(dataloader)

        return features, labels

    def _extract_features(self, dataloader):
        features, labels = [], []
        with torch.no_grad():
            for inputs, labs, _ in tqdm(dataloader, desc="Extracting Features"):
                inputs = inputs.to(self.device)
                '''
                outputs = self.model(inputs).squeeze()
                '''
                outputs = self.model(inputs)
                features.append(outputs.cpu())
                labels.append(labs)
        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        return features, labels

    @staticmethod
    def _save_features(features, labels, file_name):
        """Saves extracted features to file."""
        hf = h5py.File(file_name, 'w')
        hf.create_dataset('features', data=features)
        hf.create_dataset('labels', data=labels)
        hf.close()

    @staticmethod
    def _load_features(file_name):
        """Loads features from file."""
        hf = h5py.File(file_name, 'r')
        features = np.array(hf.get("features"))
        labels = np.array(hf.get("labels"))
        hf.close()

        return features, labels
