import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset


test_tfm = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])


class InfDataset(Dataset):
    def __init__(self, path, tfm=test_tfm):
        super(Dataset).__init__()
        self.path = path
        
        self.files = sorted([os.path.join(path, x)
                            for x in os.listdir(path) if x.endswith(".png")])
        self.filenames = [file for file in os.listdir(path)]
        self.filenames.sort()
        self.transform = tfm
        print(f"One {path} sample", self.files[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        return im
