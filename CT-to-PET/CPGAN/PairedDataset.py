#@title PairedDataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


data_transforms = {
    'Train': transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    'Test': transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    'Val': transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
}

class PairedDataset(Dataset):
    def __init__(self, root, mode):
        """
        root must have trainA trainB testA testB as its subfolders
        mode must be either 'train' or 'test'
        """
        assert mode in 'Train Test Val'.split(), 'mode should be either train or test'
        
        super().__init__()
        self.root = root
        self.mode = mode
        self.transforms = data_transforms[mode]
        # Paths
        pathA = os.path.join(self.root, mode,"A")
        pathB = os.path.join(self.root, mode,"B")
        # List of images
        dirA = os.listdir(pathA)
        dirB = os.listdir(pathB)
        # Sort
        dirA.sort()
        dirB.sort()
        
        assert len(dirA) == len(dirB), f"Unequal amount of images"
        assert dirA == dirB, "Image names are not identical"
               
        self.dirA = dirA
        self.dirB = dirB
        print(f'Found {len(self.dirA)} images of {mode}A and {len(self.dirB)} images of {mode}B')
        
    def __len__(self):
        return len(self.dirA)
    
    def load_image(self, subfolder, path):
        full_path = os.path.join(self.root, self.mode, subfolder, path)
        image = Image.open(full_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return full_path, image
    
    def __getitem__(self, idx): #doesnt support slices, we dont want them
        # we use serial batching
        pathA, imgA = self.load_image("A",self.dirA[idx])
        pathB, imgB = self.load_image("B",self.dirB[idx])
        return {
            'A': imgA, 'pathA': pathA,
            'B': imgB, 'pathB': pathB
        }