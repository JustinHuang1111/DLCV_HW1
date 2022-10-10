
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as TF
import torch


# 2a
vgg16 = models.vgg16(pretrained=True)
for param in vgg16.features.parameters():
    param.requires_grad = False


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

# 2b


class Deeplab(nn.Module):
    def __init__(self):
        super(Deeplab, self).__init__()
        self.deeplab = models.segmentation.deeplabv3_resnet50(num_classes=7)
        for m in self.modules():
            if isinstance(m, nn.Linear or nn.Conv2d):
                m.weight.data = nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x):
        out = self.deeplab(x)["out"]
        return out


class EnsembledModel(nn.Module):
    def __init__(self, state1, state2, device):
        super(EnsembledModel, self).__init__()
        self.model1 = Deeplab()
        self.model2 = Deeplab()

        self.model1.load_state_dict(torch.load(state1, map_location="cpu"))
        self.model2.load_state_dict(torch.load(state2, map_location="cpu"))

        self.model1 = self.model1.to(device)
        self.model2 = self.model2.to(device)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)

        return TF.resize((x1 + x2) / 2, 512)
