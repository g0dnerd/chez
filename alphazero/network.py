"""
AlphaZero neural network architecture.

Input: 8x8x119 planes (encoded board state with history)
Output:
  - Policy: 8x8x73 logits (move probabilities)
  - Value: scalar in [-1, 1] (expected game outcome)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoding import TOTAL_PLANES  # 119
from .policy import POLICY_PLANES  # 73


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero-style neural network.

    Architecture:
    - Input convolution: 119 -> num_filters
    - Residual tower: num_blocks residual blocks
    - Policy head: num_filters -> 73 planes
    - Value head: num_filters -> scalar

    Args:
        num_filters: Number of filters in conv layers (default: 128)
        num_blocks: Number of residual blocks (default: 6)
    """

    def __init__(self, num_filters: int = 128, num_blocks: int = 6):
        super().__init__()

        self.num_filters = num_filters
        self.num_blocks = num_blocks

        # Input convolution
        self.conv_input = nn.Conv2d(TOTAL_PLANES, num_filters, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Conv2d(32, POLICY_PLANES, 1)

        # Value head (use 32 filters like policy head for sufficient capacity)
        self.value_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 119, 8, 8)

        Returns:
            policy: Logits of shape (batch, 73, 8, 8) or (batch, 4672)
            value: Scalar of shape (batch, 1) in range [-1, 1]
        """
        # Input block
        out = F.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for block in self.residual_blocks:
            out = block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = self.policy_fc(policy)  # (batch, 73, 8, 8)
        # Reshape to (batch, 8, 8, 73) then flatten to (batch, 4672) for move indexing
        policy = policy.permute(0, 2, 3, 1).reshape(-1, 64 * POLICY_PLANES)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(-1, 32 * 64)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inference mode prediction (no gradients).

        Args:
            x: Input tensor of shape (batch, 119, 8, 8)

        Returns:
            policy: Softmax probabilities of shape (batch, 4672)
            value: Scalar of shape (batch, 1) in range [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self(x)
            policy = F.softmax(policy_logits, dim=1)
        return policy, value


def create_network(
    num_filters: int = 128,
    num_blocks: int = 6,
    device: str | torch.device | None = None
) -> AlphaZeroNetwork:
    """
    Create and initialize an AlphaZero network.

    Args:
        num_filters: Number of conv filters (64-256 typical)
        num_blocks: Number of residual blocks (4-20 typical)
        device: Device to place model on

    Returns:
        Initialized network on specified device
    """
    if device is None:
        device = get_device()
    model = AlphaZeroNetwork(num_filters, num_blocks)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device (CUDA if available and compatible)."""
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works (catches sm_120 incompatibility etc)
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError as e:
            import warnings
            warnings.warn(f"CUDA available but not working, falling back to CPU: {e}")
    return torch.device("cpu")


if __name__ == "__main__":
    print("Testing network...")

    device = get_device()
    print(f"Using device: {device}")

    # Create network with small config for testing
    model = create_network(num_filters=64, num_blocks=4, device=device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, TOTAL_PLANES, 8, 8, device=device)

    policy, value = model(x)
    print(f"Policy shape: {policy.shape}")  # (4, 4672)
    print(f"Value shape: {value.shape}")    # (4, 1)

    assert policy.shape == (batch_size, 64 * POLICY_PLANES)
    assert value.shape == (batch_size, 1)
    assert value.min() >= -1 and value.max() <= 1

    # Test predict (inference mode)
    policy_probs, value = model.predict(x)
    assert policy_probs.shape == (batch_size, 64 * POLICY_PLANES)
    assert torch.allclose(policy_probs.sum(dim=1), torch.ones(batch_size, device=device), atol=1e-5)

    print(f"Policy probs sum: {policy_probs.sum(dim=1)}")  # Should be ~1.0
    print(f"Value range: [{value.min():.3f}, {value.max():.3f}]")

    # Test with actual encoded position
    print("\nTesting with real position...")
    from .bindings import State
    from .encoding import StateEncoder
    import numpy as np

    encoder = StateEncoder()
    state = State.default()
    planes = encoder.encode(state)

    # Convert to tensor
    x = torch.from_numpy(planes).unsqueeze(0).to(device)  # (1, 119, 8, 8)

    policy, value = model.predict(x)
    print(f"Starting position value: {value.item():.4f}")
    print(f"Top 5 policy probs: {policy[0].topk(5).values.tolist()}")

    print("\nAll network tests passed!")
