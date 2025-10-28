from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        # Load image
        image = Image.open(img_path).convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label