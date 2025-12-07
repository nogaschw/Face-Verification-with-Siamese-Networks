import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torchvision.models import mobilenet_v2


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10),  # (1,105,105) → (64,96,96)
            nn.BatchNorm2d(64),                # exp 2
            nn.Dropout(0.3),                   # exp 2 Dropout added
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → (64,48,48)
            
            nn.Conv2d(64, 32, kernel_size=7), # → (128,42,42)
            nn.BatchNorm2d(32),               # exp 2
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → (128,21,21)
            
            nn.Conv2d(32, 32, kernel_size=4),# → (128,18,18)
            nn.BatchNorm2d(32),               # exp 2
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → (128,9,9)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*9*9, 1024),
            nn.BatchNorm1d(1024),    # exp 2
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),    # exp 2
            nn.Dropout(0.5),        # Dropout added
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # L1 distance
        diff = torch.abs(output1 - output2)
        return self.out(diff)


class SiameseResNet18(nn.Module):
    """
    A Siamese Neural Network using pretrained ResNet-18 as feature extractor.
    The model compares embeddings using a learned similarity head.
    """
    def __init__(self, fc_output_dim=512):
        super(SiameseResNet18, self).__init__()

        # Load pretrained ResNet-18 and remove the final classification layer
        base_model = models.resnet18(pretrained=True)
        modules = list(base_model.children())[:-1]  # remove the last FC layer
        self.feature_extractor = nn.Sequential(*modules)

        self.embedding = nn.Sequential(
            nn.Linear(512, fc_output_dim),
            nn.BatchNorm1d(fc_output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.similarity_head = nn.Sequential(
            nn.Linear(fc_output_dim * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),

        )

    def forward_once(self, x):
        x = self.feature_extractor(x)  # Output shape: (B, 512, 1, 1)
        x = torch.flatten(x, 1)        # Flatten to (B, 512)
        x = self.embedding(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        combined = torch.cat([
            out1,
            out2,
            torch.abs(out1 - out2),
            out1 * out2
        ], dim=1)

        return self.similarity_head(combined)


class SiameseResNet18(nn.Module):
    def __init__(self, fc_output_dim=512):
        super(SiameseResNet18, self).__init__()

        base_model = models.resnet18(pretrained=True)

        # Convert conv1 to accept grayscale (1 channel)
        conv1 = base_model.conv1
        base_model.conv1 = nn.Conv2d(
            3, conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None
        )
        with torch.no_grad():
            base_model.conv1.weight[:] = conv1.weight.mean(dim=1, keepdim=True)

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Remove FC

        self.embedding = nn.Sequential(
            nn.Linear(512, fc_output_dim),
            nn.BatchNorm1d(fc_output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)        # → (B, 512)
        x = self.embedding(x)          # → (B, fc_output_dim)
        x = F.normalize(x, p=2, dim=1)    
        return x

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        return emb1, emb2


class SiameseMobileNetV2(nn.Module):
    def __init__(self, fc_output_dim=256):
        super(SiameseMobileNetV2, self).__init__()
        base_model = mobilenet_v2(pretrained=True)
        self.feature_extractor = base_model.features  # (B, 1280, 7, 7)

        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, fc_output_dim),
            nn.BatchNorm1d(fc_output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.similarity_head = nn.Sequential(
            nn.Linear(fc_output_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.embedding(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        combined = torch.cat([out1, out2, torch.abs(out1 - out2), out1 * out2], dim=1)
        return self.similarity_head(combined)
