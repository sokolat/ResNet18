import argparse
import random

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--val_dir", type=str, default="./data/val")
    parser.add_argument("--test_dir", type=str, default="./data/test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = Block(64, 64)
        self.block2 = Block(64, 64)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.block3 = Block(128, 128)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.block4 = Block(256, 256)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.block5 = Block(512, 512)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=2)
        self.fc = nn.Linear(in_features=512, out_features=1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = self.conv2(x)
        x = x + self.conv3(x)
        x = x + self.block3(x)
        x = self.conv4(x)
        x = x + self.conv5(x)
        x = x + self.block4(x)
        x = self.conv6(x)
        x = x + self.conv7(x)
        x = x + self.block5(x)
        x = self.avg_pool(x)
        x = self.fc(torch.flatten(x, start_dim=1))
        return x


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        flat_data = torch.flatten(data, start_dim=1)
        cov = torch.cov(flat_data)
        eig_vals, eig_vectors = torch.linalg.eig(cov)
        alphas = torch.normal(mean=0, std=0.1, size=(3,))
        multipliers = alphas * eig_vals
        aug = eig_vectors @ multipliers
        aug = torch.unsqueeze(torch.unsqueeze(aug.real, 1), 1)

        data += aug

        return data, target

    def __len__(self):
        return len(self.dataset)


def get_size():
    return random.randint(256, 480)


def init_weight(model):
    for name, params in model.named_parameters():
        if "bias" not in name and "bn" not in name:
            torch.nn.init.kaiming_normal_(params.data)


def get_transform(size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def train(args, model, device):

    transform = get_transform(get_size())
    train_dataset = ImageFolder(root=args.train_dir, transform=transform)
    train_dataset = DatasetWrapper(train_dataset)
    val_dataset = ImageFolder(root=args.val_dir, transform=transform)
    val_dataset = DatasetWrapper(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    model.to(device)
    init_weight(model)

    optimizer = SGD(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max')

    for epoch in range(1, args.epochs + 1):

        running_loss = 0.0
        running_accuracy = 0.0

        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            running_accuracy += (output.argmax(dim=1) == target).sum().item()

            if idx % 10 == 9:
                print(
                    f"Epoch [{epoch}/{args.epochs}], Step [{idx + 1}/{len(train_loader)}], "
                    f"Loss: {running_loss / ((idx + 1) * data.size(0)):.4f}, "
                    f"Accuracy: {running_accuracy / ((idx + 1) * data.size(0)):.4f}"
                )
                breakpoint()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = running_accuracy / len(train_dataset)

        val_loss, val_accuracy = validate(model, val_loader, device, criterion)

        scheduler.step(val_accuracy)

        print(f"Epoch [{epoch}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

@torch.no_grad()
def validate(model, val_loader, device, criterion):

    running_loss = 0.0
    running_accuracy = 0.0

    model.eval()
    for data, target in val_loader:

        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        running_loss += loss.item() * data.size(0)
        running_accuracy += (output.argmax(dim=1) == target).sum().item()
    
    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = running_accuracy / len(val_loader.dataset)

    return val_loss, val_accuracy

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ResNet18()
    train(args, model, device)


if __name__ == "__main__":
    main()
