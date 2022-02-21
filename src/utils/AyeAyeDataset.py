import os
import torch
import torch.utils.data
from PIL import Image
import pandas as pd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AyeAyeDataset(torch.utils.data.Dataset):
    """Aye-aye Dataset."""
    def __init__(self, root: str, data_annotations: str, data_frames: str, transforms=None):
        """
        Args:
            root (string): Path to the root directory where images/csv files are located.
            data_annotations (string): Path to the CSV file of the images and labels.
            data_frames (string): Relative path from root to the frames directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        # Transforms are any image modifications
        self.transforms = transforms
        # CSV File of the images and labels
        self.imgs_frame = pd.read_csv(data_annotations)
        # Path to the CSV File
        self.path_to_data_annotations = data_annotations
        # Relative path to the data folder
        self.path_to_data_frames = data_frames

    def __getitem__(self, index):
        """Support indexing to get the images, labels, and names of samples"""
        # Open Image and Resize
        image = Image.open(os.path.join(self.root, self.path_to_data_frames, self.imgs_frame.iloc[index, 1])).convert("RGB").resize((100, 100))
        # Get label from the data frame for this index
        label = self.imgs_frame.iloc[index, 2]
        label = 1 if label == 3 else 0
        if self.transforms is not None:
            image = self.transforms(image)
        # Find the File Name
        name = self.imgs_frame.iloc[index, 1]
        return image, label, name

    def __len__(self):
        """Enables len(dataset) to return the length of the dataset."""
        return len(self.imgs_frame)