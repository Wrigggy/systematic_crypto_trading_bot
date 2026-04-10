# Model Inference Plugin

ONNX/PyTorch model inference for alpha generation.

## When to use

Use when your alpha requires sequential pattern recognition that
expression trees cannot capture (e.g., LSTM on 240-candle sequences).

Expression trees are preferred for most use cases because they are
self-contained, auditable, and require no binary checkpoints.

## Usage

Enable in config:
```yaml
plugins:
  model_inference:
    enabled: true
```
