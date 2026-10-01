import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")

    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 2
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
        print(f"epoch {epoch}  training loss {total_loss / seen:.4f}")

    os.makedirs("models", exist_ok=True)
    path = "models/model_fp32.pth"
    torch.save(model, path)
    print(f"Saved FP32 model to {path}")


if __name__ == "__main__":
    main()
