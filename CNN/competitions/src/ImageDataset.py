from PIL import Image
from torch.utils.data import Dataset
import os

class ImageDataset(Dataset):
    def __init__(self, data_path, dataframe, transform=None, test_mode=False):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.data_path = data_path
        self.test_mode = test_mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        img_path = os.path.join(self.data_path, img_path)
        
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        if self.test_mode:
            file_path = self.data.iloc[idx, 0]
            return image, file_path
        else:
            label = self.data.iloc[idx, 1]
            return image, label