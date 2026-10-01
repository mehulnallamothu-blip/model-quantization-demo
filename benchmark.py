import os
import time

import torch
import torchao.quantization  # noqa: F401  registers Int8Tensor so the INT8 model can load
from torchvision import datasets, transforms


def accuracy(model, images, labels, batch_size):
    correct = 0
    with torch.no_grad():
        for start in range(0, images.size(0), batch_size):
            batch_images = images[start : start + batch_size]
            batch_labels = labels[start : start + batch_size]
            predictions = model(batch_images).argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
    return 100.0 * correct / images.size(0)


def inference_seconds(model, images, batch_size, repeats):
    # One warmup pass, then the average of several full passes.
    # Timing depends on this computer and the PyTorch build.
    with torch.no_grad():
        for start in range(0, images.size(0), batch_size):
            model(images[start : start + batch_size])

        times = []
        for _ in range(repeats):
            start_time = time.perf_counter()
            for start in range(0, images.size(0), batch_size):
                model(images[start : start + batch_size])
            times.append(time.perf_counter() - start_time)
    return sum(times) / len(times)


def load_model(path):
    model = torch.load(path, weights_only=False, map_location="cpu")
    model.eval()
    return model


def main():
    torch.manual_seed(0)
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    images = torch.stack([image for image, _label in test_data])
    labels = torch.tensor([label for _image, label in test_data])

    fp32_path = "models/model_fp32.pth"
    int8_path = "models/model_int8.pth"
    fp32_model = load_model(fp32_path)
    int8_model = load_model(int8_path)

    batch_size = 64
    repeats = 5
    results = {
        "FP32": {
            "accuracy": accuracy(fp32_model, images, labels, batch_size),
            "size": os.path.getsize(fp32_path),
            "seconds": inference_seconds(fp32_model, images, batch_size, repeats),
        },
        "INT8": {
            "accuracy": accuracy(int8_model, images, labels, batch_size),
            "size": os.path.getsize(int8_path),
            "seconds": inference_seconds(int8_model, images, batch_size, repeats),
        },
    }

    accuracy_change = results["INT8"]["accuracy"] - results["FP32"]["accuracy"]
    size_ratio = results["INT8"]["size"] / results["FP32"]["size"]
    time_ratio = results["INT8"]["seconds"] / results["FP32"]["seconds"]

    text = f"""FP32 Model
Accuracy: {results["FP32"]["accuracy"]:.2f}%
Model size: {results["FP32"]["size"]} bytes
Inference time: {results["FP32"]["seconds"]:.4f} s average over {repeats} passes of {images.size(0)} test images

INT8 Model
Accuracy: {results["INT8"]["accuracy"]:.2f}%
Model size: {results["INT8"]["size"]} bytes
Inference time: {results["INT8"]["seconds"]:.4f} s average over {repeats} passes of {images.size(0)} test images

Difference:
Accuracy change (INT8 - FP32): {accuracy_change:+.2f} percentage points
Model size: INT8 file is {size_ratio:.2f}x the FP32 file
Inference time: INT8 took {time_ratio:.2f}x as long as FP32 in this run

Measured on CPU. One warmup pass was run before timing.
Inference time depends on the computer and the software environment.
"""

    os.makedirs("results", exist_ok=True)
    out_path = "results/comparison.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text, end="")
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
