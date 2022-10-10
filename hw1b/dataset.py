import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import os
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset


def transform(image: Image):
    image = TF.resize(image, 400)
    image = TF.to_tensor(image)
    return image


class InfDataset(Dataset):
    def __init__(
        self,
        path: str,
    ):
        # test files with xxx.png
        self.images_list = sorted(
            [os.path.join(path, x)
             for x in os.listdir(path) if x.endswith(".jpg")]
        )
        self.filenames = [file for file in os.listdir(
            path) if file.endswith(".jpg")]
        self.filenames.sort()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        sats = transform(Image.open(self.images_list[idx]))

        return sats
