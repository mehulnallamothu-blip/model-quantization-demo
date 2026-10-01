import os

import torch
from torch import nn
from torchvision import datasets, transforms
from torchao.quantization import Int8StaticActivationInt8WeightConfig, quantize_
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_primitives import MappingType

# INT8 range used for the activation scale and zero-point.
QMIN = -128
QMAX = 127


def activation_scale_and_zero_point(xmin, xmax):
    # Map the observed float range onto INT8:
    # real_value = scale * (int8_value - zero_point)
    scale = (xmax - xmin) / (QMAX - QMIN)
    if scale == 0:
        scale = 1.0
    zero_point = int(round(QMIN - xmin / scale))
    zero_point = min(QMAX, max(QMIN, zero_point))
    return scale, zero_point


def collect_activation_ranges(model, loader, max_samples):
    ranges = {}

    def hook(name):
        def record(_module, inputs, _output):
            values = inputs[0].detach()
            xmin = values.min().item()
            xmax = values.max().item()
            if name not in ranges:
                ranges[name] = [xmin, xmax]
            else:
                ranges[name][0] = min(ranges[name][0], xmin)
                ranges[name][1] = max(ranges[name][1], xmax)

        return record

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook(name)))

    seen = 0
    with torch.no_grad():
        for images, _labels in loader:
            model(images)
            seen += images.size(0)
            if seen >= max_samples:
                break

    for handle in handles:
        handle.remove()
    return ranges, seen


def main():
    device = torch.device("cpu")
    model = torch.load("models/model_fp32.pth", weights_only=False, map_location=device)
    model.to(device)
    model.eval()

    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64)
    ranges, seen = collect_activation_ranges(model, train_loader, max_samples=1024)
    print(f"Calibrated on {seen} MNIST training images")

    # Activations: one scale and zero-point per linear layer (per tensor).
    # Weights: torchao stores each output row with its own symmetric scale.
    # Bias, Flatten, and ReLU stay in FP32.
    for name, (xmin, xmax) in ranges.items():
        scale, zero_point = activation_scale_and_zero_point(xmin, xmax)
        print(
            f"linear {name}: activation scale {scale:.6f}, "
            f"zero-point {zero_point} (input range {xmin:.4f} to {xmax:.4f})"
        )
        config = Int8StaticActivationInt8WeightConfig(
            act_quant_scale=torch.tensor([[scale]], dtype=torch.float32),
            act_quant_zero_point=torch.tensor([[zero_point]], dtype=torch.int8),
            granularity=[PerTensor(), PerRow()],
            act_mapping_type=MappingType.ASYMMETRIC,
            set_inductor_config=False,
        )

        def match_this_layer(_module, fqn, layer_name=name):
            return fqn == layer_name

        quantize_(model, config, filter_fn=match_this_layer)

    # torchao installs an extra_repr that cannot be pickled.
    for module in model.modules():
        module.__dict__.pop("extra_repr", None)

    os.makedirs("models", exist_ok=True)
    path = "models/model_int8.pth"
    torch.save(model, path)
    print(f"Saved INT8 model to {path}")
    print("Linear weights are INT8. Bias, Flatten, and ReLU remain FP32.")


if __name__ == "__main__":
    main()
