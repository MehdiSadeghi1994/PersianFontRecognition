import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image


class FontData(Dataset):
    
    def __init__ (self, root_path, transform=None, for_data=None):
        super(FontData, self).__init__() 

        self.transform = transform
        self.images_add = []
        self.for_data = for_data

        for image_name in os.listdir(root_path):
            current_add = os.path.join(root_path, image_name)
            self.images_add.append(current_add)

    def __len__(self):
        return len(self.images_add)


    def __str__(self):
        return f'{self.for_data} DatSet \nNumber of Samples is: {len(self)}\n'


    def __getitem__(self, idx):

        # image = torch.from_numpy(np.asarray(Image.open(self.images_add[idx]))).float().div(255)
        image = Image.open(self.images_add[idx])
        if self.transform:
            image = self.transform(image)
        image = torch.from_numpy(np.asarray(image)).float().div(255)
        image_name = os.path.basename(os.path.normpath(self.images_add[idx]))
        label = int(image_name.split('_')[0])

        return image, label





if __name__ == '__main__':
    data_path = '/content/drive/MyDrive/Font_Recognition/Data/Train'
    dataset = FontData(root_path=data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=10)

    for image, label in dataloader:
        print(image.shape)
        print(label)

        break









