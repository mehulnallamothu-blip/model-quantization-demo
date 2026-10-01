import os

import torch
from torchvision import datasets, transforms


def main():
    path = "models/model_fp32.pth"
    device = torch.device("cpu")
    model = torch.load(path, weights_only=False, map_location=device)
    model.to(device)
    model.eval()

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    size_bytes = os.path.getsize(path)
    print(f"test accuracy: {accuracy:.2f}%")
    print(f"test samples: {total}")
    print(f"model size: {size_bytes} bytes")


if __name__ == "__main__":
    main()
