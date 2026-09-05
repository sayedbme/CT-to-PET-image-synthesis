#@title JoinedPairedDataset.py

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
        pathA = os.path.join(self.root, mode)
        # List of images
        dirA = os.listdir(pathA)
        # Sort
        dirA.sort()

        self.dirA = dirA
        self.dirB = dirA
        print(f'Found {len(self.dirA)} images of {mode} in Joint Format')
        
    def __len__(self):
        return len(self.dirA)
    
    def split_image(self, full_image):
        # split AB image into A and B
        w, h = full_image.size
        w2 = int(w / 2)
        A = full_image.crop((0, 0, w2, h))
        B = full_image.crop((w2, 0, w, h))
        if self.transforms:
            A = self.transforms(A)
            B = self.transforms(B)
        return A, B
    
    def __getitem__(self, idx): #doesnt support slices, we dont want them
        # we use serial batching
        full_path = os.path.join(self.root, self.mode, self.dirA[idx])
        full_image = Image.open(full_path).convert("RGB")
        # Split Images
        imgA, imgB = self.split_image(full_image)
        pathA = pathB = full_path # same path for all
        return {
            'A': imgA, 'pathA': pathA,
            'B': imgB, 'pathB': pathB
        }