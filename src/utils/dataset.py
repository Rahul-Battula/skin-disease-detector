import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dir1, img_dir2, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.transform = transform

        # Map class names to integer labels
        self.classes = sorted(self.df['dx'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, image_id):
        path1 = os.path.join(self.img_dir1, image_id + '.jpg')
        path2 = os.path.join(self.img_dir2, image_id + '.jpg')
        return path1 if os.path.exists(path1) else path2

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._get_image_path(row['image_id'])
        image = Image.open(img_path).convert('RGB')

        label = self.class_to_idx[row['dx']]

        if self.transform:
            image = self.transform(image)

        return image, label