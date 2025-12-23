# StarNet EASE Backbone Documentation

## Overview

This document describes the StarNet backbone implementation for the EASE incremental learning framework.

## Features

- **Full Adapter Support**: Compatible with EASE's adapter-based incremental learning
- **Multiple Variants**: Three model sizes available (S1, S2, Base)
- **Flexible Input Size**: Optimized for 64x64 images (configurable)
- **Prototype Learning**: Implements `forward_proto` method for prototype calculation

## Model Variants

### StarNet-S1 (Small)
- Embedding dimensions: [32, 64, 128, 256]
- Depths: [2, 2, 6, 2]
- Output dimension: 256
- Best for: Quick experiments, limited computational resources

### StarNet-S2 (Medium)
- Embedding dimensions: [64, 128, 256, 512]
- Depths: [2, 2, 8, 2]
- Output dimension: 512
- Best for: Balanced performance and efficiency

### StarNet-Base
- Embedding dimensions: [96, 192, 384, 768]
- Depths: [3, 3, 9, 3]
- Output dimension: 768
- Best for: Maximum performance

## Usage

### Configuration

Update your experiment JSON file to use StarNet:

```json
{
  "backbone_type": "starnet_s1_ease",
  "model_name": "ease",
  "ffn_num": 64,
  ...
}
```

Available backbone types:
- `starnet_s1_ease`
- `starnet_s2_ease`
- `starnet_base_ease`

### Example: Training with StarNet

```python
from utils.inc_net import get_backbone
from easydict import EasyDict

args = {
    "backbone_type": "starnet_s1_ease",
    "model_name": "ease",
    "ffn_num": 64,
    "device": ["cuda:0"]
}

# Create the backbone
backbone = get_backbone(args)

# The backbone is ready to use with EASE framework
```

## Architecture Details

### Convolutional Backbone

StarNet uses a convolutional architecture instead of transformers:

1. **Stem Layer**: 4x4 convolution with stride 4 for initial feature extraction
2. **Multiple Stages**: Each stage contains multiple StarNetBlocks
3. **Adapter Integration**: Adapters can be inserted at each block
4. **Global Average Pooling**: Final feature aggregation

### Adapter Mechanism

The adapter mechanism works the same way as ViT EASE:

```python
# Get new adapter for a new task
backbone.get_new_adapter()

# Train on the new task...

# After training, freeze and add to adapter list
backbone.freeze()
backbone.add_adapter_to_list()
```

### Forward Modes

#### Training Mode
```python
features = backbone.forward(x, test=False)
# Returns: (batch_size, out_dim)
```

#### Testing Mode
```python
features = backbone.forward(x, test=True, use_init_ptm=False)
# Returns: (batch_size, out_dim * num_adapters)
# Concatenates features from all adapters
```

#### Prototype Mode
```python
features = backbone.forward_proto(x, adapt_index=0)
# Returns: (batch_size, out_dim)
# Uses specified adapter for feature extraction
```

## Method Reference

### Core Methods

#### `forward_proto(x, adapt_index)`
Extracts features for prototype calculation.

**Parameters:**
- `x` (Tensor): Input images, shape (B, 3, H, W)
- `adapt_index` (int): 
  - `-1`: Use initial PTM without adapters
  - `0 to len(adapter_list)-1`: Use historical adapter
  - `>= len(adapter_list)`: Use current adapter

**Returns:**
- Feature tensor of shape (B, out_dim)

#### `get_new_adapter()`
Creates and initializes a new adapter for the current task.

#### `add_adapter_to_list()`
Saves the current adapter to history and creates a new one.

#### `freeze()`
Freezes all backbone parameters except the current adapter.

## Input Size Considerations

The default configuration is optimized for 64x64 images:

- **Stem**: 4x4 conv with stride 4 → reduces 64x64 to 16x16
- **Stage transitions**: Further downsampling via stride-2 convolutions
- **Final feature map**: Small spatial size suitable for global pooling

For different input sizes, you may need to adjust:
- Stem kernel size and stride
- Number of downsampling stages
- Stage depths

## Comparison with ViT EASE

| Feature | ViT EASE | StarNet EASE |
|---------|----------|--------------|
| Architecture | Transformer | CNN |
| Input processing | Patch embedding | Stem convolution |
| Feature extraction | CLS token | Global avg pooling |
| Adapter insertion | FFN blocks | Conv blocks |
| Best for | Global context | Local patterns |
| Input size | 224x224 | 64x64 (flexible) |

## Troubleshooting

### Common Issues

1. **"AttributeError: forward_proto"**
   - Solution: Ensure you're using the updated code with starnet_ease.py

2. **"Unknown type starnet_s1_ease"**
   - Solution: Check that utils/inc_net.py has been updated with StarNet support

3. **Out of memory**
   - Solution: Try smaller variant (S1) or reduce batch size

4. **Poor performance**
   - Solution: Adjust `ffn_num` (adapter bottleneck size)
   - Try different model variants

## Performance Tips

1. **Adapter Size**: 
   - Start with `ffn_num=64`
   - Increase for better capacity, decrease for efficiency

2. **Model Selection**:
   - Use S1 for datasets with simple patterns
   - Use Base for complex, high-resolution features

3. **Batch Size**:
   - StarNet is more memory-efficient than ViT
   - Can use larger batch sizes

## Citation

If you use StarNet EASE in your research, please cite:

```bibtex
@inproceedings{starnet,
  title={StarNet: Efficient Convolutional Architecture for Computer Vision},
  author={...},
  booktitle={...},
  year={2024}
}

@article{ease,
  title={EASE: Efficient Adapter-based Subspace Encoding for Incremental Learning},
  author={...},
  journal={...},
  year={2024}
}
```

## Support

For issues or questions:
1. Check this documentation
2. Review the example configurations in `exps/` directory
3. Open an issue on GitHub with:
   - Configuration file
   - Error message
   - Environment details
