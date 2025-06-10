# ResNet18 Implementation for ImageNette Classification

This project implements a ResNet18 model for image classification using the ImageNette dataset, which is a subset of ImageNet containing 10 easily distinguishable classes. The implementation includes advanced features such as Weights & Biases logging, TensorBoard integration, and custom data augmentation.

## Features

- ResNet18 architecture implementation from scratch
- Custom data augmentation using eigenvector-based perturbation
- Integration with Weights & Biases for experiment tracking
- TensorBoard support for local visualization
- Checkpoint saving and loading for training resumption
- Learning rate scheduling with ReduceLROnPlateau
- Random seed control for reproducibility

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- tensorboard
- wandb
- numpy
- matplotlib

You can install the dependencies using pipenv:

```bash
pipenv install
```

## Project Structure

```
.
├── resnet18.py          # Main implementation file
├── data/                # Dataset directory
│   ├── train/           # Training data
│   └── val/            # Validation data
├── checkpoints/         # Model checkpoints
├── runs/               # TensorBoard logs
└── wandb/              # Weights & Biases logs
```

## Usage

To train the model with default parameters:

```bash
python resnet18.py
```

### Command Line Arguments

- `--device`: Device to use for training (default: "cuda")
- `--train_dir`: Training data directory (default: "./data/train")
- `--val_dir`: Validation data directory (default: "./data/val")
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.1)
- `--weight_decay`: Weight decay for optimizer (default: 0.0001)
- `--momentum`: Momentum for optimizer (default: 0.9)
- `--epochs`: Number of training epochs (default: 30)
- `--wandb_project_name`: Weights & Biases project name (default: "resnet18")
- `--wandb_entity`: Weights & Biases entity/username
- `--track`: Enable Weights & Biases tracking
- `--checkpoint`: Path to checkpoint file for resuming training
- `--seed`: Random seed for reproducibility (default: 42)

### Examples

Train with Weights & Biases tracking:
```bash
python resnet18.py --track --wandb_entity your_username
```

Resume training from checkpoint:
```bash
python resnet18.py --checkpoint checkpoints/1234567890/checkpoint_epoch_10.pth
```

## Model Architecture

The ResNet18 implementation follows the original architecture with:
- Initial 7x7 convolution layer
- 4 layers of residual blocks (2 blocks each)
- Average pooling and final fully connected layer
- Total of 18 layers with learnable parameters

## Data Augmentation

The implementation includes a custom data augmentation technique using eigenvector-based perturbation:
1. Flattens input images
2. Computes covariance matrix
3. Performs eigendecomposition
4. Applies random perturbations along eigenvectors

## Monitoring and Logging

- **TensorBoard**: Training metrics are logged to the `runs/` directory
- **Weights & Biases**: When enabled, syncs with TensorBoard and provides additional experiment tracking

## License

This project is open source and available under the MIT License.
