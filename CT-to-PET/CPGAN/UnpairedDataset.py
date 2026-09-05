import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

#@title UnpairedDataset.py
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


class ImageFolder:
    def __init__(self, root, transforms):
        self.root = root
        self.paths = os.listdir(root)
        self.images = []
        self.transforms = transforms
            
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx): #doesnt support slices, we dont want them
        idx = idx % len(self)
        full_path = os.path.join(self.root, self.paths[idx])
        image = Image.open(full_path).convert("RGB")
        im_shape = np.array(image).shape
        if len(im_shape) !=3:
            print(full_path, im_shape)
        return full_path, self.transforms(image)
    

class UnpairedDataset(Dataset):
    def __init__(self, root, mode):
        """
        root must have trainA trainB testA testB as its subfolders
        mode must be either 'train' or 'test'
        """
        assert mode in 'Train Test Val'.split(), 'mode should be either train or test'
        
        super().__init__()
        self.transforms = data_transforms[mode]

        pathA = os.path.join(root, mode,"A")
        self.dirA = ImageFolder(pathA, self.transforms)

        pathB = os.path.join(root, mode,"B")
        self.dirB = ImageFolder(pathB, self.transforms) 
        
        print(f'Found {len(self.dirA)} images of {mode}A and {len(self.dirB)} images of {mode}B')
        
    def __len__(self):
        return max(len(self.dirA), len(self.dirB))
    
    def load_image(self, path):
        image = Image.open(path)
        if self.transforms:
            image = self.transforms(image)
        return path, image
    
    def __getitem__(self, idx): #doesnt support slices, we dont want them
        # we use serial batching
        pathA, imgA = self.dirA[idx]
        pathB, imgB = self.dirB[idx]
        return {
            'A': imgA, 'pathA': pathA,
            'B': imgB, 'pathB': pathB
        }