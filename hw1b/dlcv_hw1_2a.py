# training code

import imageio.v2 as imageio
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import models
import imageio
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random
myseed = 1314520  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""# Preparing Dataset

"""

config = {
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "image_size": 512,
    "batch_size": 8,
    "exp_name": "FCN32",
    "epoch": 50,
    "ckpt_dir": "DL.ckpt"
}


def read_masks(file_list):
    n_masks = len(file_list)
    masks = torch.empty(
        (n_masks, 512, 512), dtype=torch.int)

    for i, file in enumerate(file_list):
        mask = imageio.imread(file)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
        masks[i, mask == 2] = 3  # (Green: 010) Forest land
        masks[i, mask == 1] = 4  # (Blue: 001) Water
        masks[i, mask == 7] = 5  # (White: 111) Barren land
        masks[i, mask == 0] = 6  # (Black: 000) Unknown
    # print(masks.shape)
    return masks


def transform(image: Image, mask: torch.Tensor, aug: bool):
    mask = TF.to_pil_image(mask)

    if random.random() > 0.5 and aug:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5 and aug:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    return image, torch.squeeze(mask)


class Dataset(Dataset):
    def __init__(
        self,
        path: str,
        mode: str,
    ):
        self.images_list = sorted(
            [os.path.join(path, x)
             for x in os.listdir(path) if x.endswith(".jpg")]
        )

        labels_list = sorted(
            [os.path.join(path, x)
             for x in os.listdir(path) if x.endswith(".png")]
        )
        self.mode = mode
        print("reading masks...")
        self.labels = read_masks(labels_list)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        sats, masks = transform(Image.open(
            self.images_list[idx]), self.labels[idx], aug=(self.mode == "train"))

        return sats, masks


_dataset_dir = "./hw1/hw1_data/p2_data"
train_set = Dataset(
    path=os.path.join(_dataset_dir, "train"), mode="train"
)
valid_set = Dataset(
    path=os.path.join(_dataset_dir, "validation"), mode="validation"
)
train_loader = DataLoader(train_set, shuffle=True,
                          batch_size=config["batch_size"])
valid_loader = DataLoader(valid_set, shuffle=True,
                          batch_size=config["batch_size"])
print(len(train_set))
print(len(valid_set))


_exp_name = config["exp_name"]


vgg16 = models.vgg16(pretrained=True)
for param in vgg16.features.parameters():
    param.requires_grad = False


class Unet(nn.Module):

    def __init__(self):
        super().__init__()

        self.unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=3, out_channels=7, init_features=32, pretrained=False)

    def forward(self, x):

        return self.unet(x)


class Deeplab(nn.Module):

    def __init__(self):
        super().__init__()

        self.deeplab = models.segmentation.deeplabv3_resnet50(
            weights=None, num_classes=7)
        self.deeplab.classifier[4] = nn.Conv2d(256, 7, kernel_size=1, stride=1)
        # self.deeplab.aux_classifier[4] = nn.Conv2d(
        #     256, 7, kernel_size=1, stride=1)

    def forward(self, x):

        return self.deeplab(x)['out']


class FCN32(nn.Module):
    def __init__(self):
        super(FCN32, self).__init__()
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(4096, 21, 1),
            nn.ConvTranspose2d(21, 21, 224, stride=32)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
"""# Training

"""


def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    # print("label", labels[0])
    # print("pred", pred[0])
    mean_iou = 0
    for i in range(6):
        tp_fp = torch.sum(pred == i)
        tp_fn = torch.sum(labels == i)
        tp = torch.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp + 1e-16)
        mean_iou = mean_iou + iou / 6
    return mean_iou


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        tensor = tensor.to("cpu")
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(
            1, tensor.view(n, 1, h, w).to(torch.int64), 1)
        return one_hot.to(device)

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


device = "cuda" if torch.cuda.is_available() else "cpu"

drive_dir = "./hw1/"

# "cuda" only when GPUs are available.

# The number of training epochs and patience.
n_epochs = config["epoch"]
patience = 300  # If no improvement in 'patience' epochs, early stop

# Initialize a model, and put it on the device specified.

model = FCN32().to(device)
# model = models.vgg16_bn(pretrained=False).to(device)
# For the classification task, we use cross-entropy as the measurement of performance.

criterion_pre = nn.CrossEntropyLoss()
criterion_post = SoftIoULoss(7)


# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(
    model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-5)
# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0


to_iou = 1000
# model.load_state_dict(torch.load(config["ckpt_dir"]))

# checkpoint = torch.load(config["ckpt_dir"])
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# init_epoch = checkpoint['epoch']
# loss = checkpoint['loss']
init_epoch = 0

for epoch in range(init_epoch, n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()
        # print(imgs.shape,labels.shape)

        # print(np.shape(imgs.to(device)[0]))
        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        if epoch < to_iou:
            labels = labels.squeeze(1).to(device, dtype=torch.long)
            loss = criterion_pre(logits, labels)
        else:
            labels = labels.squeeze(1).to(device)
            loss = criterion_post(logits, labels)
        # loss = criterion(logits, labels)
        # loss =  nn.CrossEntropyLoss(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = mean_iou_score(logits.argmax(dim=1), labels)
        # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(
        f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        # loss = -1 * criterion(logits, labels.to(device))
        if epoch < to_iou:
            labels = labels.squeeze(1).to(device, dtype=torch.long)
            loss = criterion_pre(logits, labels)
        else:
            labels = labels.squeeze(1).to(device)
            loss = criterion_post(logits, labels)

        # Compute the accuracy for current batch.

        acc = mean_iou_score(logits.argmax(dim=1), labels)

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        # break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    scheduler.step(valid_loss)

    # Print the information.
    print(
        f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt", "a") as f:
            print(
                f"[ Valid {_exp_name} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            f.write(
                f"[ Valid {_exp_name} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best\n[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}\n")

    else:
        with open(f"./{_exp_name}_log.txt", "a") as f:
            print(
                f"[ Valid {_exp_name} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            f.write(
                f"[ Valid {_exp_name} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}\n[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}\n")

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        # only save best to prevent output memory exceed error
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(drive_dir, f"{_exp_name}.pt"))
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(
                f"No improvment {patience} consecutive epochs, early stopping")
            break
    if epoch == 3 or epoch % 10 == 0:
        torch.save(model.state_dict(), f"{_exp_name}_best_{epoch}.ckpt")
