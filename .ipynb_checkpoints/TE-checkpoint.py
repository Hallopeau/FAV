import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix

class Tester:
    def __init__(self, model, test_loader, dataset):
        self.model = model
        self.test_loader = test_loader
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(self):

        self.model.eval()
        
        df = pd.DataFrame(columns=['target', 'output'])

        with torch.no_grad():
            for j, (images, labels) in enumerate(self.test_loader):
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)

                for i in range(outputs.shape[0]): 

                    label_i = labels[i]
                    output_i = outputs[i]

                    id = int(j*10000 + i)
                    
                    _, predicted = torch.max(output_i.data, 0)

                    if label_i == self.dataset.classes.index("favelas"):
                        df.loc[id, "target"] = 1
                    if label_i == self.dataset.classes.index("residential"):
                        df.loc[id, "target"] = 0
                    
                    if predicted == self.dataset.classes.index("favelas"):
                        df.loc[id, "output"] = 1
                    if predicted == self.dataset.classes.index("residential"):
                        df.loc[id, "output"] = 0
    
        target_array = df['target'].to_numpy().astype(np.int64)
        output_array = df['output'].to_numpy().astype(np.int64)

        IoU = jaccard_score(target_array, output_array)
        cm = confusion_matrix(target_array, output_array)
        df_cm = pd.DataFrame(cm, index=['Actual Class 0', 'Actual Class 1'], columns=['Predicted Class 0', 'Predicted Class 1'])
        cr = classification_report(target_array, output_array)

        self.report = [cm, cr, IoU]

        print(f"\nJaccard index: {IoU*100: 0.1f}%\n")
        print(f"\n{df_cm}\n")
        print(f"\n{cr}\n")
