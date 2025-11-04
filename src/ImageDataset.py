from PIL import Image
from torch.utils.data import Dataset
import os

class ImageDataset(Dataset):
    def __init__(self, data_path, dataframe, transform=None, test_mode=False):
        """
        Dataset for loading images.
        
        Args:
            data_path: Base path to the data directory
            dataframe: DataFrame with image paths (and optionally labels)
            transform: Optional transforms to apply
            test_mode: If True, expects only file paths (no labels) and returns file path instead of label
        """
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.data_path = data_path
        self.test_mode = test_mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image path (always in first column)
        img_path = self.data.iloc[idx, 0]
        img_path = os.path.join(self.data_path, img_path)
        
        # Load image
        image = Image.open(img_path).convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.test_mode:
            # For test mode, return the relative file path instead of a label
            file_path = self.data.iloc[idx, 0]
            return image, file_path
        else:
            # For train/validation mode, return the label
            label = self.data.iloc[idx, 1]
            return image, label