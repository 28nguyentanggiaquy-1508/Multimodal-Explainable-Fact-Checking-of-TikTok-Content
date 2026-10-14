import math
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Utility: output length for a 1-D convolution
# ─────────────────────────────────────────────────────────────────────────────

def calc_out_dims_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: int,
) -> Tuple[int, int, int]:
    """Return (L_out, batch_size, n_channels) for a 1-D convolution.

    Formula mirrors PyTorch's Conv1d:
        L_out = floor((L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    """
    batch_size, n_channels, L = x.shape
    L_out = math.floor(
        (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )
    return L_out, batch_size, n_channels


# ─────────────────────────────────────────────────────────────────────────────
# Core 1-D KAN convolution engine
# ─────────────────────────────────────────────────────────────────────────────

def multiple_convs_kan_conv1d(
    x: torch.Tensor,           # (batch, n_channels, L)
    kernels: list,             # list of KAN_Convolution_1D objects
    kernel_size: int,
    out_channels: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    """Apply multiple KAN 1-D convolutions and accumulate into output channels.

    Strategy:
      - Treat the 1-D signal as a degenerate 2-D image of shape (L, 1) so we
        can reuse torch.nn.Unfold with kernel (kernel_size, 1).
      - This keeps dilation and padding support identical to the 2-D version.

    Args:
        x            : Input tensor  (batch, n_channels, L).
        kernels      : List of KAN_Convolution_1D modules, length = in_channels * out_channels.
        kernel_size  : Length of the 1-D kernel window.
        out_channels : Number of output feature maps.
        stride       : Convolution stride.
        dilation     : Kernel dilation.
        padding      : Zero-padding added to both ends of the sequence.
        device       : Torch device string.

    Returns:
        Tensor of shape (batch, out_channels, L_out).
    """
    L_out, batch_size, n_channels = calc_out_dims_1d(x, kernel_size, stride, dilation, padding)

    # Lift (batch, C, L) → (batch, C, L, 1) so nn.Unfold can work
    x_2d = x.unsqueeze(-1)  # (batch, n_channels, L, 1)

    unfold = nn.Unfold(
        kernel_size=(kernel_size, 1),
        dilation=(dilation, 1),
        padding=(padding, 0),
        stride=(stride, 1),
    )
    # unfolded: (batch, n_channels * kernel_size, L_out)
    unfolded = unfold(x_2d)

    # Reshape → (batch, n_channels, L_out, kernel_size)
    conv_groups = (
        unfolded
        .view(batch_size, n_channels, kernel_size, L_out)
        .permute(0, 1, 3, 2)          # (batch, n_channels, L_out, kernel_size)
        .contiguous()
    )

    matrix_out = torch.zeros((batch_size, out_channels, L_out), device=device)
    kern_per_out = len(kernels) // out_channels

    for c_out in range(out_channels):
        accum = torch.zeros((batch_size, L_out), device=device)
        for k_idx in range(kern_per_out):
            kernel = kernels[c_out * kern_per_out + k_idx]
            # Flatten batch & position dims → (batch*L_out, kernel_size)
            patches = conv_groups[:, k_idx, :, :].flatten(0, 1)
            # KANLinear maps kernel_size → 1; reshape back to (batch, L_out)
            result = kernel.conv.forward(patches).view(batch_size, L_out)
            accum += result
        matrix_out[:, c_out, :] = accum

    return matrix_out


# ─────────────────────────────────────────────────────────────────────────────
# KANLinear  (unchanged from the 2-D version — operates on flat feature vectors)
# ─────────────────────────────────────────────────────────────────────────────

class KANLinear(nn.Module):
    """B-spline based learnable linear layer (Kolmogorov–Arnold Network)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight   = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise   = scale_noise
        self.scale_base    = scale_base
        self.scale_spline  = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output   = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step + x_sorted[0] - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat([
            grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
            grid,
            grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
        ], dim=0)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced))

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        l1_fake = self.spline_weight.abs().mean(-1)
        reg_activation = l1_fake.sum()
        p = l1_fake / reg_activation
        reg_entropy = -torch.sum(p * p.log())
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy


# ─────────────────────────────────────────────────────────────────────────────
# KAN (stacked KANLinear layers) — unchanged
# ─────────────────────────────────────────────────────────────────────────────

class KAN(nn.Module):
    """Stack of KANLinear layers forming a fully connected KAN."""

    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.layers = nn.ModuleList([
            KANLinear(i, o, grid_size, spline_order, scale_noise, scale_base,
                      scale_spline, True, base_activation, grid_eps, grid_range)
            for i, o in zip(layers_hidden, layers_hidden[1:])
        ])

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        return sum(
            l.regularization_loss(regularize_activation, regularize_entropy)
            for l in self.layers
        )


# ─────────────────────────────────────────────────────────────────────────────
# KAN_Convolution_1D  —  single 1-D KAN kernel
# ─────────────────────────────────────────────────────────────────────────────

class KAN_Convolution_1D(nn.Module):
    """Single 1-D KAN convolutional kernel.

    Internally holds one KANLinear that maps a flattened 1-D patch
    (length = kernel_size) to a scalar output.

    Args:
        kernel_size  : Length of the sliding window.
        stride       : Step between consecutive windows.
        padding      : Zero-padding added to both ends.
        dilation     : Spacing between kernel elements.
        grid_size    : KANLinear grid resolution.
        spline_order : B-spline order.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.grid_size    = grid_size
        self.spline_order = spline_order

        self.conv = KANLinear(
            in_features=kernel_size,
            out_features=1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, 1, L)  →  (batch, 1, L_out)"""
        return multiple_convs_kan_conv1d(
            x, [self], self.kernel_size, 1,
            self.stride, self.dilation, self.padding, x.device
        )

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        return self.conv.regularization_loss(regularize_activation, regularize_entropy)


# ─────────────────────────────────────────────────────────────────────────────
# KAN_Convolutional_Layer_1D  —  multi-channel 1-D KAN conv layer
# ─────────────────────────────────────────────────────────────────────────────

class KAN_Convolutional_Layer_1D(nn.Module):
    """1-D KAN Convolutional Layer with in_channels × out_channels kernels.

    Mirrors KAN_Convolutional_Layer from the 2-D code but operates on
    sequences of shape (batch, in_channels, L).

    Args:
        in_channels  : Number of input feature maps / embedding dimensions.
        out_channels : Number of output feature maps.
        kernel_size  : Length of the 1-D sliding window.
        stride       : Convolution stride.
        padding      : Zero-padding on both sequence ends.
        dilation     : Kernel dilation factor.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation

        # One KAN_Convolution_1D per (in_channel, out_channel) pair
        self.convs = nn.ModuleList([
            KAN_Convolution_1D(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            for _ in range(in_channels * out_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, in_channels, L)  →  (batch, out_channels, L_out)"""
        self.device = x.device
        return multiple_convs_kan_conv1d(
            x, self.convs, self.kernel_size, self.out_channels,
            self.stride, self.dilation, self.padding, self.device
        )

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        return sum(
            c.regularization_loss(regularize_activation, regularize_entropy)
            for c in self.convs
        )


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper: flat KAN head (re-exported for convenience)
# ─────────────────────────────────────────────────────────────────────────────

class FlattenKANWrapper(nn.Module):
    """Flatten any input and pass it through a KAN MLP."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (128, 64), output_dim: int = 1):
        super().__init__()
        self.kan = KAN(layers_hidden=[input_dim, *hidden_dims, output_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x.reshape(x.size(0), -1), update_grid=False)


# ─────────────────────────────────────────────────────────────────────────────
# Ready-to-use 1-D KAN model variants  (text / sequence classification)
# ─────────────────────────────────────────────────────────────────────────────
#
# Expected input shape: (batch, in_channels, seq_len)
# For plain text with a pre-computed embedding, set in_channels = embed_dim
# and seq_len = number of tokens.
#
# ─────────────────────────────────────────────────────────────────────────────

class KANC_MLP_1D(nn.Module):
    """Small 1-D KAN model — analogous to KANC_MLP.

    Two KAN conv layers + MaxPool1d + single linear head.

    Args:
        in_channels  : Input channels (e.g. embedding dimension).
        num_classes  : Number of output classes.
        grid_size    : KANLinear grid resolution.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer_1D(
            in_channels=in_channels, out_channels=5, kernel_size=3, grid_size=grid_size
        )
        self.conv2 = KAN_Convolutional_Layer_1D(
            in_channels=5, out_channels=5, kernel_size=3, grid_size=grid_size
        )
        self.pool1   = nn.MaxPool1d(kernel_size=2)
        self.flat    = nn.Flatten()
        self.linear1 = nn.LazyLinear(num_classes)
        self.name    = f"KANC MLP 1D (Small) (gs={grid_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, in_channels, seq_len)"""
        x = self.pool1(self.conv1(x))
        x = self.pool1(self.conv2(x))
        x = self.flat(x)
        x = self.linear1(x)
        return F.log_softmax(x, dim=1)


class KANC_MLP_1D_Medium(nn.Module):
    """Medium 1-D KAN model — analogous to KANC_MLP_Medium.

    Two KAN conv layers + MaxPool1d + two-layer linear head.

    Args:
        in_channels  : Input channels.
        num_classes  : Number of output classes.
        hidden_dim   : Hidden units in the MLP head.
        grid_size    : KANLinear grid resolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_dim: int = 256,
        grid_size: int = 5,
    ):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer_1D(
            in_channels=in_channels, out_channels=5, kernel_size=3, grid_size=grid_size
        )
        self.conv2 = KAN_Convolutional_Layer_1D(
            in_channels=5, out_channels=10, kernel_size=3, grid_size=grid_size
        )
        self.pool1   = nn.MaxPool1d(kernel_size=2)
        self.flat    = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.name    = f"KANC MLP 1D (Medium) (gs={grid_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool1(self.conv2(x))
        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


class KANC_MLP_1D_Big(nn.Module):
    """Large 1-D KAN model — analogous to KANC_MLP_Big.

    Two KAN conv layers + MaxPool1d + KAN MLP head (learnable activations).

    Args:
        in_channels  : Input channels.
        num_classes  : Number of output classes.
        grid_size    : KANLinear grid resolution.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer_1D(
            in_channels=in_channels, out_channels=5, kernel_size=3, grid_size=grid_size
        )
        self.conv2 = KAN_Convolutional_Layer_1D(
            in_channels=5, out_channels=10, kernel_size=3, grid_size=grid_size
        )
        self.pool1  = nn.MaxPool1d(kernel_size=2)
        self.flat   = nn.Flatten()
        # LazyLinear infers the flattened size automatically
        self.linear1 = nn.LazyLinear(128)
        self.linear2 = nn.Linear(128, num_classes)
        self.name    = f"KANC MLP 1D (Big) (gs={grid_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, in_channels, seq_len)
           Also accepts (batch, seq_len, embed_dim) and auto-transposes.
        """
        # Accept (batch, seq_len, embed_dim) layout common in NLP pipelines
        if x.dim() == 3 and x.size(1) != self.conv1.in_channels:
            x = x.transpose(1, 2)          # → (batch, embed_dim, seq_len)

        x = self.pool1(self.conv1(x))
        x = self.pool1(self.conv2(x))
        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


class KANC_Regression_1D(nn.Module):
    """1-D KAN model for sequence regression (scalar output).

    Useful for tasks like sentiment scoring, time-series forecasting, etc.

    Args:
        in_channels : Input channels.
        grid_size   : KANLinear grid resolution.
    """

    def __init__(self, in_channels: int = 1, grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer_1D(
            in_channels=in_channels, out_channels=5, kernel_size=3, grid_size=grid_size
        )
        self.conv2 = KAN_Convolutional_Layer_1D(
            in_channels=5, out_channels=10, kernel_size=3, grid_size=grid_size
        )
        self.pool1   = nn.MaxPool1d(kernel_size=2)
        self.flat    = nn.Flatten()
        self.linear1 = nn.LazyLinear(100)
        self.linear2 = nn.Linear(100, 1)
        self.name    = f"KANC Regression 1D (gs={grid_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) != self.conv1.in_channels:
            x = x.transpose(1, 2)
        x = self.pool1(self.conv1(x))
        x = self.pool1(self.conv2(x))
        x = self.flat(x)
        x = self.linear1(x)
        return self.linear2(x)          # raw scalar, no activation


# ─────────────────────────────────────────────────────────────────────────────
# Quick usage example (runs when executed as a script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    seq_len    = 64     # e.g. token sequence length
    embed_dim  = 16     # e.g. character embedding or shallow word embedding

    # Input: (batch, embed_dim, seq_len)  ← channels-first convention
    x = torch.randn(batch_size, embed_dim, seq_len).to(device)

    print("KANC_MLP_1D (Small)")
    model = KANC_MLP_1D(in_channels=embed_dim, num_classes=5, grid_size=5).to(device)
    out = model(x)
    print(f"  input : {x.shape}  →  output : {out.shape}")   # (4, 5)

    print("\nKANC_MLP_1D_Medium")
    model = KANC_MLP_1D_Medium(in_channels=embed_dim, num_classes=5, grid_size=5).to(device)
    out = model(x)
    print(f"  input : {x.shape}  →  output : {out.shape}")

    print("\nKANC_MLP_1D_Big")
    model = KANC_MLP_1D_Big(in_channels=embed_dim, num_classes=5, grid_size=5).to(device)
    out = model(x)
    print(f"  input : {x.shape}  →  output : {out.shape}")

    print("\nKANC_Regression_1D")
    model = KANC_Regression_1D(in_channels=embed_dim, grid_size=5).to(device)
    out = model(x)
    print(f"  input : {x.shape}  →  output : {out.shape}")   # (4, 1)

    print("\nAll forward passes OK.")
