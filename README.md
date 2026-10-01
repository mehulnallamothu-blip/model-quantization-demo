# model-quantization-demo

A small project that trains an MNIST classifier in FP32, makes an INT8 copy with post-training quantization, and compares accuracy, file size, and inference time.

## What quantization means

A neural network normally stores numbers as 32-bit floating point (FP32). Quantization converts many of those numbers to a smaller integer type, here 8-bit integers (INT8). The integer is only a code. A scale, and sometimes a zero-point, turn that code back into an approximate real number:

```
FP32 value
→ scale and zero-point
→ INT8 value
```

The formula used for activations in this project is:

```
real_value ≈ scale * (int8_value - zero_point)
```

`scale` is the step size between integer codes. `zero_point` is the integer code that stands for the real number 0. Together they place the 256 INT8 codes over the range of values seen in the data. Values that fall between two codes are rounded, so quantization is an approximation and accuracy can change a little.

Fewer bits usually means a smaller file. An FP32 weight is 4 bytes. An INT8 weight is 1 byte. Scales and zero-points are stored too, so the file is not exactly one quarter of the original size.

## What this project does

`train.py` trains a small network for 2 epochs on MNIST:

Flatten → Linear → ReLU → Linear

Training uses FP32 on CPU. The model is saved to `models/model_fp32.pth`.

`quantize.py` does post-training quantization. The trained model is left as it is. The script runs 1024 MNIST training images through it and records the minimum and maximum activation into each linear layer. From that range it computes one scale and one zero-point per layer, then asks torchao to store the linear weights as INT8.

PyTorch 2.14 marks the older `torch.ao.quantization` helpers as deprecated, so this project uses torchao's `Int8StaticActivationInt8WeightConfig`:

- Linear weights are INT8. Each output row has its own symmetric weight scale.
- Linear inputs are quantized to INT8 with the calibrated per-tensor scale and zero-point.
- Bias, Flatten, and ReLU stay in FP32.

The quantized model is saved to `models/model_int8.pth`.

`evaluate.py` prints FP32 test accuracy, the number of test images, and the FP32 file size.

`benchmark.py` runs both models on the same 10,000 MNIST test images. It warms up once, times 5 full passes on CPU, and writes `results/comparison.txt`.

## Results from this run

These numbers come from running the scripts locally. They are not a general claim about INT8.

| | FP32 | INT8 |
| --- | --- | --- |
| Test accuracy | 96.07% | 96.09% |
| File size | 410875 bytes | 109429 bytes |
| Inference time | 0.0079 s | 0.0662 s |

The INT8 file is 0.27x the FP32 file. Accuracy changed by +0.02 percentage points. On this Mac CPU, the INT8 model took 8.38x as long as the FP32 model.

INT8 is not always faster. The matrix multiply here is INT8, but the model is tiny, so the time spent quantizing activations can be larger than the time saved. A different computer, a larger model, or a different kernel can change that result.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python evaluate.py
python quantize.py
python benchmark.py
```

MNIST is downloaded into `data/` the first time you train. That folder is gitignored.

## Limitations

- The network is intentionally small, and it is only trained for 2 epochs.
- Calibration uses 1024 training images, not the full training set.
- Only the linear layers are quantized. Bias and ReLU stay in FP32.
- Timing is from one CPU run on this machine. Another machine will differ.
