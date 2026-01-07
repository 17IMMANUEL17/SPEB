"""
REPRODUCIBILITY METADATA & INSTRUCTIONS
---------------------------------------
For strict bit-exact reproducibility (ICML standard), you must match the environment.
Code determinism alone is insufficient across different hardware architectures due
to non-associative floating-point arithmetic.

1. Dependencies:
   Ensure you use exact library versions. Run:
   pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 numpy==1.26.3 matplotlib==3.8.2

2. Hardware Note:
   This code is deterministic. However, running on different GPU architectures
   (e.g., RTX 3090 vs A100) will yield slightly different trajectories due to
   hardware-level implementation differences in cuBLAS.
   
   To reproduce the exact numbers reported, use: [Insert Your GPU Here, e.g., RTX 4090]
"""

import os
import random
import sys

# -----------------------------
# 0. STRICT REPRODUCIBILITY SETUP
# -----------------------------
# CRITICAL: Must be set before ANY torch imports.
# Enables deterministic algorithms in cuBLAS (used by PyTorch for matmul/conv).
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.grad import conv2d_weight
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass

# -----------------------------
# 1. PRECISION & HARDWARE CONTROL
# -----------------------------
# CRITICAL: Disable TensorFloat-32 (TF32) on Ampere+ GPUs (A100, RTX 3090/4090, H100).
# TF32 is non-deterministic relative to standard FP32 and causes silent precision loss.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# Ensure standard float32 precision
torch.set_default_dtype(torch.float32)

# -----------------------------
# 2. SEEDING & WORKER SETUP
# -----------------------------
def seed_everything(seed=42):
    """
    Sets seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
    # Ensure deterministic behavior in CuDNN
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Force deterministic algorithms. 
    # warn_only=False ensures strict compliance (crashes if an op is non-deterministic)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except AttributeError:
        # Fallback for older PyTorch versions
        torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    """
    Worker initialization function for DataLoader to ensure each worker 
    operates with a deterministic seed derived from the base seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def print_env_info():
    """Prints environment info for reproducibility logs."""
    print("\n" + "="*40)
    print("REPRODUCIBILITY ENVIRONMENT CHECK")
    print("="*40)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"TF32 Allowed (Matmul): {torch.backends.cuda.matmul.allow_tf32}")
        print(f"TF32 Allowed (CuDNN): {torch.backends.cudnn.allow_tf32}")
    print("="*40 + "\n")

# -----------------------------
# Activations & Model
# -----------------------------
def relu(u): return torch.relu(u)
def relu_prime(u): return (u > 0).to(u.dtype)

@dataclass
class CNN9:
    # Block 1: 32x32 -> 16x16
    W1: torch.Tensor; b1: torch.Tensor   # (64, 3, 3, 3)   s=1
    W2: torch.Tensor; b2: torch.Tensor   # (64, 64, 3, 3)  s=2 (pool)
    
    # Block 2: 16x16 -> 8x8
    W3: torch.Tensor; b3: torch.Tensor   # (128, 64, 3, 3) s=1
    W4: torch.Tensor; b4: torch.Tensor   # (128, 128, 3, 3) s=2 (pool)
    
    # Block 3: 8x8 -> 4x4
    W5: torch.Tensor; b5: torch.Tensor   # (256, 128, 3, 3) s=1
    W6: torch.Tensor; b6: torch.Tensor   # (256, 256, 3, 3) s=2 (pool)
    
    # Block 4: 4x4 -> 2x2
    W7: torch.Tensor; b7: torch.Tensor   # (512, 256, 3, 3) s=1
    W8: torch.Tensor; b8: torch.Tensor   # (512, 512, 3, 3) s=2 (pool)
    
    # Classifier: 2x2 -> Flat
    W9: torch.Tensor; b9: torch.Tensor   # (10, 512*2*2)

    @property
    def device(self): return self.W1.device

# -----------------------------
# Forward & Gradients
# -----------------------------
@torch.no_grad()
def forward_u_sig(net: CNN9, x0, m):
    u = [None] * 9
    sig = [None] * 9
    
    # Block 1
    u[0] = F.conv2d(x0, net.W1, net.b1, stride=1, padding=1);   sig[0] = relu(u[0])
    u[1] = F.conv2d(m[0], net.W2, net.b2, stride=2, padding=1); sig[1] = relu(u[1])
    
    # Block 2
    u[2] = F.conv2d(m[1], net.W3, net.b3, stride=1, padding=1); sig[2] = relu(u[2])
    u[3] = F.conv2d(m[2], net.W4, net.b4, stride=2, padding=1); sig[3] = relu(u[3])
    
    # Block 3
    u[4] = F.conv2d(m[3], net.W5, net.b5, stride=1, padding=1); sig[4] = relu(u[4])
    u[5] = F.conv2d(m[4], net.W6, net.b6, stride=2, padding=1); sig[5] = relu(u[5])
    
    # Block 4
    u[6] = F.conv2d(m[5], net.W7, net.b7, stride=1, padding=1); sig[6] = relu(u[6])
    u[7] = F.conv2d(m[6], net.W8, net.b8, stride=2, padding=1); sig[7] = relu(u[7])
    
    # FC
    B = x0.shape[0]
    m8_flat = m[7].reshape(B, -1)
    u[8] = m8_flat @ net.W9.t() + net.b9
    sig[8] = u[8] 

    return u, sig

class XZState:
    def __init__(self, B, device):
        self.dims = [
            (B, 64, 32, 32), (B, 64, 16, 16),
            (B, 128, 16, 16), (B, 128, 8, 8),
            (B, 256, 8, 8),   (B, 256, 4, 4),
            (B, 512, 4, 4),   (B, 512, 2, 2),
            (B, 10)
        ]
        self.x = [torch.zeros(d, device=device) for d in self.dims]
        self.z = [torch.zeros(d, device=device) for d in self.dims]

    def reset(self, B, device):
        if self.x[0].shape[0] != B:
            self.__init__(B, device)

@torch.no_grad()
def xz_relax_batch_grad(
    net: CNN9, x0, y,
    eta=1.0, K=25,
    state: XZState | None = None,
    tol: float = 1e-4,
    warm_start: bool = True,
    beta: float = 1.0,
    collect_convergence_metrics: bool = False,
):
    device = net.device
    B = x0.shape[0]
    num_layers = 9
    
    # Label Smoothing
    eps_ls = 0.1
    y_onehot = F.one_hot(y, num_classes=10).to(x0.dtype)
    y_smooth = (1.0 - eps_ls) * y_onehot + eps_ls / 10.0

    if state is None:
        state = XZState(B, device)
    else:
        state.reset(B, device)
    
    x = state.x; z = state.z
    
    convergence_history = [] if collect_convergence_metrics else None
    
    steps_taken = 0
    for _ in range(K):
        steps_taken += 1
        
        m = [(xi + zi) * 0.5 for xi, zi in zip(x, z)]
        s = [(xi - zi) for xi, zi in zip(x, z)]
        
        u, sig = forward_u_sig(net, x0, m)
        F_err = [(si - mi) for si, mi in zip(sig, m)]
        
        p = torch.softmax(m[8], dim=1)
        g_last = (p - y_smooth)
        
        q = [None] * num_layers
        q[8] = s[8] 
        for i in range(7, -1, -1):
            q[i] = relu_prime(u[i]) * s[i]

        Jt = [None] * num_layers
        Jt[8] = -s[8] 
        WTq8 = (q[8] @ net.W9).reshape(B, 512, 2, 2)
        Jt[7] = -s[7] + WTq8
        Jt[6] = -s[6] + F.conv_transpose2d(q[7], net.W8, stride=2, padding=1, output_padding=1)
        Jt[5] = -s[5] + F.conv_transpose2d(q[6], net.W7, stride=1, padding=1)
        Jt[4] = -s[4] + F.conv_transpose2d(q[5], net.W6, stride=2, padding=1, output_padding=1)
        Jt[3] = -s[3] + F.conv_transpose2d(q[4], net.W5, stride=1, padding=1)
        Jt[2] = -s[2] + F.conv_transpose2d(q[3], net.W4, stride=2, padding=1, output_padding=1)
        Jt[1] = -s[1] + F.conv_transpose2d(q[2], net.W3, stride=1, padding=1)
        Jt[0] = -s[0] + F.conv_transpose2d(q[1], net.W2, stride=2, padding=1, output_padding=1)

        total_change = 0.0
        max_change = 0.0
        # Hidden Layers
        for i in range(8):
            dx = F_err[i] + 0.5 * Jt[i]
            dz = F_err[i] - 0.5 * Jt[i]
            x[i].add_(dx, alpha=eta)
            z[i].add_(dz, alpha=eta)
            change = dx.abs().mean().item()
            total_change += change
            max_change = max(max_change, change)
            
        # Output Layer
        dx8 = F_err[8] + 0.5 * Jt[8] + 0.5 * beta * g_last
        dz8 = F_err[8] - 0.5 * Jt[8] - 0.5 * beta * g_last
        x[8].add_(dx8, alpha=eta)
        z[8].add_(dz8, alpha=eta)
        change8 = dx8.abs().mean().item()
        total_change += change8
        max_change = max(max_change, change8)

        if collect_convergence_metrics:
            fixedpoint_residuals = [F_err[i].abs().mean().item() for i in range(9)]
            convergence_history.append({
                'iteration': steps_taken,
                'total_change': total_change,
                'max_change': max_change,
                'fixedpoint_residuals': fixedpoint_residuals,
                'avg_fixedpoint_residual': sum(fixedpoint_residuals) / len(fixedpoint_residuals)
            })

        if total_change < tol:
            break

    # Compute Gradients
    m = [(xi + zi) * 0.5 for xi, zi in zip(x, z)]
    s = [(xi - zi) for xi, zi in zip(x, z)]
    u, _ = forward_u_sig(net, x0, m)
    
    delta = [None] * num_layers
    delta[8] = s[8]
    for i in range(8):
        delta[i] = relu_prime(u[i]) * s[i]

    gradsW = []
    gradsb = []

    # Manual conv gradients
    gradsW.append(conv2d_weight(x0, net.W1.shape, delta[0], stride=1, padding=1) / B)
    gradsb.append(delta[0].sum(dim=(0,2,3)) / B)
    
    gradsW.append(conv2d_weight(m[0], net.W2.shape, delta[1], stride=2, padding=1) / B)
    gradsb.append(delta[1].sum(dim=(0,2,3)) / B)
    
    gradsW.append(conv2d_weight(m[1], net.W3.shape, delta[2], stride=1, padding=1) / B)
    gradsb.append(delta[2].sum(dim=(0,2,3)) / B)
    
    gradsW.append(conv2d_weight(m[2], net.W4.shape, delta[3], stride=2, padding=1) / B)
    gradsb.append(delta[3].sum(dim=(0,2,3)) / B)
    
    gradsW.append(conv2d_weight(m[3], net.W5.shape, delta[4], stride=1, padding=1) / B)
    gradsb.append(delta[4].sum(dim=(0,2,3)) / B)
    
    gradsW.append(conv2d_weight(m[4], net.W6.shape, delta[5], stride=2, padding=1) / B)
    gradsb.append(delta[5].sum(dim=(0,2,3)) / B)
    
    gradsW.append(conv2d_weight(m[5], net.W7.shape, delta[6], stride=1, padding=1) / B)
    gradsb.append(delta[6].sum(dim=(0,2,3)) / B)
    
    gradsW.append(conv2d_weight(m[6], net.W8.shape, delta[7], stride=2, padding=1) / B)
    gradsb.append(delta[7].sum(dim=(0,2,3)) / B)
    
    m7_flat = m[7].reshape(B, -1)
    gradsW.append((delta[8].t() @ m7_flat) / B)
    gradsb.append(delta[8].mean(dim=0))

    ce = F.cross_entropy(m[8], y).item()
    
    if collect_convergence_metrics:
        return tuple(gradsW), tuple(gradsb), ce, steps_taken - 1, convergence_history
    else:
        return tuple(gradsW), tuple(gradsb), ce, steps_taken - 1

# -----------------------------
# Autograd Reference
# -----------------------------
def autograd_grads_like_cnn9(net: CNN9, x, y):
    params = {}
    for i in range(1, 10):
        params[f"W{i}"] = getattr(net, f"W{i}").detach().clone().requires_grad_(True)
        params[f"b{i}"] = getattr(net, f"b{i}").detach().clone().requires_grad_(True)
    
    h = x
    h = relu(F.conv2d(h, params['W1'], params['b1'], stride=1, padding=1))
    h = relu(F.conv2d(h, params['W2'], params['b2'], stride=2, padding=1))
    h = relu(F.conv2d(h, params['W3'], params['b3'], stride=1, padding=1))
    h = relu(F.conv2d(h, params['W4'], params['b4'], stride=2, padding=1))
    h = relu(F.conv2d(h, params['W5'], params['b5'], stride=1, padding=1))
    h = relu(F.conv2d(h, params['W6'], params['b6'], stride=2, padding=1))
    h = relu(F.conv2d(h, params['W7'], params['b7'], stride=1, padding=1))
    h = relu(F.conv2d(h, params['W8'], params['b8'], stride=2, padding=1))
    logits = h.reshape(x.size(0), -1) @ params['W9'].t() + params['b9']
    
    loss = F.cross_entropy(logits, y, label_smoothing=0.1)
    loss.backward()
    
    gradsW = tuple(params[f"W{i}"].grad for i in range(1, 10))
    gradsb = tuple(params[f"b{i}"].grad for i in range(1, 10))
    return gradsW, gradsb, float(loss.detach())

# -----------------------------
# Metrics & Plotting
# -----------------------------
def flat_cat(tup):
    return torch.cat([t.reshape(-1) for t in tup], dim=0)

def cos_sim(a, b, eps=1e-12):
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = (a_flat.norm() * b_flat.norm()).clamp_min(eps)
    return float((a_flat @ b_flat) / denom)

def relative_error(a, b, eps=1e-12):
    return float((a - b).norm() / b.norm().clamp_min(eps))

def compute_bias_metrics(gradsW_est, gradsW_true):
    bias_per_layer = []
    for i in range(len(gradsW_est)):
        diff = gradsW_est[i] - gradsW_true[i]
        bias = diff.mean().item()
        bias_per_layer.append(bias)
    return bias_per_layer

def compute_variance_metrics(gradsW_est, gradsW_true):
    var_per_layer = []
    for i in range(len(gradsW_est)):
        diff = gradsW_est[i] - gradsW_true[i]
        var = diff.var().item()
        var_per_layer.append(var)
    return var_per_layer

def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "lines.linewidth": 2.5,
        "figure.figsize": (8, 6),
    })

def plot_results_icml(results, eta_values):
    set_style()
    plt.figure()
    
    styles = {
        0.25: {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': r'$\eta=0.25$'},
        0.50: {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'label': r'$\eta=0.50$'},
        0.75: {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'label': r'$\eta=0.75$'},
        1.00: {'color': '#d62728', 'marker': 'D', 'linestyle': ':', 'label': r'$\eta=1.00$'},
    }
    
    for eta in eta_values:
        if eta not in results: continue
        data = results[eta]
        cos_steps = data['cos_steps']
        rel_err = data['relerr_globalW_hist']
        if len(cos_steps) == 0: continue
        
        st = styles.get(eta, {'color': 'black', 'marker': 'x', 'linestyle': '-'})
        plt.plot(cos_steps, rel_err, **st)

    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel(r'Relative Grad Error')
    plt.title('Global Gradient Fidelity (CNN9)')
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cnn9_grad_fidelity.png', dpi=300)
    print("Saved cnn9_grad_fidelity.png")

def plot_convergence_icml(results, eta_values):
    set_style()
    plt.figure()
    
    styles = {
        0.25: {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': r'$\eta=0.25$'},
        0.50: {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'label': r'$\eta=0.50$'},
        0.75: {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'label': r'$\eta=0.75$'},
        1.00: {'color': '#d62728', 'marker': 'D', 'linestyle': ':', 'label': r'$\eta=1.00$'},
    }
    
    for eta in eta_values:
        if eta not in results: continue
        data = results[eta]
        avg_steps = data['avg_steps_per_epoch']
        epochs = range(1, len(avg_steps) + 1)
        
        st = styles.get(eta, {'color': 'black', 'marker': 'x', 'linestyle': '-'})
        plt.plot(epochs, avg_steps, **st)

    plt.xlabel('Epochs')
    plt.ylabel('Avg. Convergence Steps ($K$)')
    plt.title('Relaxation Convergence Speed (CNN9)')
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cnn9_convergence_steps.png', dpi=300)
    print("Saved cnn9_convergence_steps.png")

def plot_gradient_noise(results, eta_values):
    set_style()
    plt.figure()
    
    styles = {
        0.25: {'color': '#1f77b4', 'linestyle': '-', 'label': r'$\eta=0.25$'},
        0.50: {'color': '#ff7f0e', 'linestyle': '--', 'label': r'$\eta=0.50$'},
        0.75: {'color': '#2ca02c', 'linestyle': '-.', 'label': r'$\eta=0.75$'},
        1.00: {'color': '#d62728', 'linestyle': ':', 'label': r'$\eta=1.00$'},
    }
    
    for eta in eta_values:
        if eta not in results: continue
        data = results[eta]
        if 'gradient_snr' not in data: continue
        
        steps = data['cos_steps']
        snr = data['gradient_snr']
        
        st = styles.get(eta, {'color': 'black', 'linestyle': '-'})
        plt.plot(steps, snr, **st)

    plt.xlabel('Training Steps')
    plt.ylabel('Gradient SNR (Signal-to-Noise Ratio)')
    plt.title('Gradient Estimation Quality')
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cnn9_gradient_snr.png', dpi=300)
    print("Saved cnn9_gradient_snr.png")

# -----------------------------
# Helpers
# -----------------------------
@torch.no_grad()
def sgd_momentum_step(net: CNN9, gradsW, gradsb, vW, vb,
                      lr=0.01, momentum=0.9, weight_decay=5e-4, clip=1.0):
    for i in range(9):
        dWi = gradsW[i] + weight_decay * getattr(net, f"W{i+1}")
        dbi = gradsb[i]
        
        gn = (dWi.norm()**2 + dbi.norm()**2)**0.5
        scale = 1.0 if gn <= clip else (clip / (gn + 1e-12))
        dWi *= scale; dbi *= scale

        vW[i].mul_(momentum).add_(dWi)
        vb[i].mul_(momentum).add_(dbi)

        getattr(net, f"W{i+1}").sub_(lr * vW[i])
        getattr(net, f"b{i+1}").sub_(lr * vb[i])

@torch.no_grad()
def ema_update(ema_net: CNN9, net: CNN9, decay=0.999):
    for i in range(1, 10):
        for param in ["W", "b"]:
            name = f"{param}{i}"
            getattr(ema_net, name).mul_(decay).add_(getattr(net, name), alpha=(1.0 - decay))

@torch.no_grad()
def accuracy(net: CNN9, loader, device, max_batches=800):
    correct = 0; total = 0
    for i, (x,y) in enumerate(loader):
        if i >= max_batches: break
        x = x.to(device); y = y.to(device)
        
        h = x
        h = relu(F.conv2d(h, net.W1, net.b1, stride=1, padding=1))
        h = relu(F.conv2d(h, net.W2, net.b2, stride=2, padding=1))
        h = relu(F.conv2d(h, net.W3, net.b3, stride=1, padding=1))
        h = relu(F.conv2d(h, net.W4, net.b4, stride=2, padding=1))
        h = relu(F.conv2d(h, net.W5, net.b5, stride=1, padding=1))
        h = relu(F.conv2d(h, net.W6, net.b6, stride=2, padding=1))
        h = relu(F.conv2d(h, net.W7, net.b7, stride=1, padding=1))
        h = relu(F.conv2d(h, net.W8, net.b8, stride=2, padding=1))
        logits = h.reshape(x.size(0), -1) @ net.W9.t() + net.b9
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def kaiming_init(shape, device):
    fan_in = (shape[1] * shape[2] * shape[3]) if len(shape) == 4 else shape[1]
    return math.sqrt(2.0 / fan_in) * torch.randn(shape, device=device)

def cosine_lr(step, total_steps, lr_max=0.05, lr_min=5e-4):
    t = step / max(1, total_steps)
    return lr_min + 0.5*(lr_max - lr_min)*(1.0 + math.cos(math.pi * t))

class Cutout(object):
    def __init__(self, length):
        self.length = length
    def __call__(self, img):
        _, h, w = img.shape
        mask = torch.ones((h, w), dtype=torch.float32)
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img

# -----------------------------
# Main
# -----------------------------
def main():
    # 1. INITIAL SEED (Global setup)
    print_env_info()
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)
    
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
        Cutout(length=8)
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])
    
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tfm)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tfm)
    
    # --- REPRODUCIBLE DATALOADERS ---
    # We must use a separate generator for the DataLoader shuffler
    g_train = torch.Generator()
    g_train.manual_seed(42)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=64, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        worker_init_fn=seed_worker,  # Ensure workers are seeded deterministically
        generator=g_train            # Ensure shuffling is deterministic
    )
    
    g_test = torch.Generator()
    g_test.manual_seed(42)
    
    test_loader  = DataLoader(
        test_ds, 
        batch_size=256, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g_test
    )

    epochs = 100
    eta_values = [0.25, 0.5, 0.75, 1.0]
    K_limit = 1000
    base_tolerance = 1e-6
    lr_max = 0.035
    lr_min = 0.0002
    compare_every = 100  
    
    results = {}

    for eta in eta_values:
        # Adaptive tolerance logic
        current_tol = 9e-5 if eta >= 0.9 else base_tolerance
        
        print(f"\n========================================")
        print(f"Training CNN9 (eta={eta}, K={K_limit}, tol={current_tol})")
        print(f"========================================")
        
        # 2. RESEED INSIDE LOOP
        # Essential: Ensures every eta experiment starts from the exact same weights
        seed_everything(42)

        W1 = kaiming_init((64, 3, 3, 3), device);   b1 = torch.zeros(64, device=device)
        W2 = kaiming_init((64, 64, 3, 3), device);  b2 = torch.zeros(64, device=device)
        W3 = kaiming_init((128, 64, 3, 3), device);  b3 = torch.zeros(128, device=device)
        W4 = kaiming_init((128, 128, 3, 3), device); b4 = torch.zeros(128, device=device)
        W5 = kaiming_init((256, 128, 3, 3), device); b5 = torch.zeros(256, device=device)
        W6 = kaiming_init((256, 256, 3, 3), device); b6 = torch.zeros(256, device=device)
        W7 = kaiming_init((512, 256, 3, 3), device); b7 = torch.zeros(512, device=device)
        W8 = kaiming_init((512, 512, 3, 3), device); b8 = torch.zeros(512, device=device)
        W9 = kaiming_init((10, 512*2*2), device);    b9 = torch.zeros(10, device=device)

        net = CNN9(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,W9,b9)
        ema_net = CNN9(W1.clone(),b1.clone(),W2.clone(),b2.clone(),W3.clone(),b3.clone(),
                       W4.clone(),b4.clone(),W5.clone(),b5.clone(),W6.clone(),b6.clone(),
                       W7.clone(),b7.clone(),W8.clone(),b8.clone(),W9.clone(),b9.clone())

        vW = [torch.zeros_like(getattr(net, f"W{i}")) for i in range(1,10)]
        vb = [torch.zeros_like(getattr(net, f"b{i}")) for i in range(1,10)]

        total_steps = epochs * len(train_loader)
        global_step = 0
        state = XZState(64, device) 
        
        # Logging lists
        cos_globalW_hist = []
        relerr_globalW_hist = []
        norm_ratio_hist = [] 
        k_steps_hist = []
        cos_steps = []
        avg_steps_per_epoch = []
        
        train_loss_hist = []
        test_acc_hist = []
        learning_rate_hist = []
        gradient_norm_hist = []
        gradient_snr = []
        bias_metrics = []
        variance_metrics = []
        early_stop_ratio = []
        layer_wise_metrics = []
        convergence_samples = []

        for ep in range(1, epochs+1):
            running_ce = 0.0
            epoch_steps_accum = 0
            num_batches = 0
            early_stops_this_epoch = 0
            
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                lr = cosine_lr(global_step, total_steps, lr_max=lr_max, lr_min=lr_min)

                collect_conv = (global_step % (compare_every * 10) == 0)
                
                result = xz_relax_batch_grad(
                    net, x, y, eta=eta, K=K_limit, state=state, 
                    tol=current_tol, warm_start=True, beta=1.0,
                    collect_convergence_metrics=collect_conv
                )
                
                if collect_conv:
                    gradsW, gradsb, ce, steps_taken, conv_history = result
                    convergence_samples.append({
                        'step': global_step,
                        'convergence_history': conv_history
                    })
                else:
                    gradsW, gradsb, ce, steps_taken = result
                
                if steps_taken < K_limit - 1:
                    early_stops_this_epoch += 1
                
                # ---- LOGGING BLOCK EVERY 100 STEPS ----
                if global_step % compare_every == 0:
                    gradsW_ag, gradsb_ag, _ = autograd_grads_like_cnn9(net, x, y)
                    
                    gx = flat_cat(gradsW)
                    ga = flat_cat(gradsW_ag)
                    
                    c_sim = cos_sim(gx, ga)
                    r_err = relative_error(gx, ga)
                    
                    n_est = gx.norm().item()
                    n_true = ga.norm().item()
                    ratio = n_est / (n_true + 1e-12)
                    
                    signal_power = (ga.norm() ** 2).item()
                    noise_power = ((gx - ga).norm() ** 2).item()
                    snr = signal_power / (noise_power + 1e-12)
                    
                    cos_globalW_hist.append(c_sim)
                    relerr_globalW_hist.append(r_err)
                    norm_ratio_hist.append(ratio)
                    k_steps_hist.append(steps_taken)
                    cos_steps.append(global_step)
                    gradient_norm_hist.append(n_true)
                    gradient_snr.append(snr)
                    
                    bias_per_layer = compute_bias_metrics(gradsW, gradsW_ag)
                    var_per_layer = compute_variance_metrics(gradsW, gradsW_ag)
                    
                    bias_metrics.append({
                        'step': global_step,
                        'layer_biases': bias_per_layer,
                        'mean_bias': sum(bias_per_layer) / len(bias_per_layer)
                    })
                    
                    variance_metrics.append({
                        'step': global_step,
                        'layer_variances': var_per_layer,
                        'mean_variance': sum(var_per_layer) / len(var_per_layer)
                    })
                    
                    current_step_layer_stats = []
                    for l_idx in range(9):
                        g_est_l = gradsW[l_idx]
                        g_true_l = gradsW_ag[l_idx]
                        
                        l_cos = cos_sim(g_est_l, g_true_l)
                        l_rel = relative_error(g_est_l, g_true_l)
                        l_true_norm = g_true_l.norm().item()
                        l_est_norm = g_est_l.norm().item()
                        
                        current_step_layer_stats.append({
                            'layer': f"W{l_idx+1}",
                            'cos_sim': l_cos,
                            'rel_err': l_rel,
                            'true_grad_norm': l_true_norm,
                            'est_grad_norm': l_est_norm
                        })
                    
                    layer_wise_metrics.append({
                        'step': global_step,
                        'layers': current_step_layer_stats
                    })
                    
                    print(f"[Step {global_step}] Global Cos: {c_sim:.4f} | Global RelErr: {r_err:.4f} | "
                          f"NormRatio: {ratio:.4f} | SNR: {snr:.2f} | Last K: {steps_taken}")

                sgd_momentum_step(net, gradsW, gradsb, vW, vb, lr=lr)
                ema_update(ema_net, net, decay=0.9995)

                global_step += 1
                running_ce += ce
                epoch_steps_accum += steps_taken
                num_batches += 1
                learning_rate_hist.append(lr)

            test_acc = accuracy(ema_net, test_loader, device)
            train_loss = running_ce / num_batches
            avg_k = epoch_steps_accum / num_batches
            early_stop_pct = early_stops_this_epoch / num_batches
            
            avg_steps_per_epoch.append(avg_k)
            train_loss_hist.append(train_loss)
            test_acc_hist.append(test_acc)
            early_stop_ratio.append(early_stop_pct)
            
            print(f">>> Ep {ep}: Loss {train_loss:.4f} | ACC {test_acc*100:.2f}% | "
                  f"Avg K: {avg_k:.1f} | Early Stop: {early_stop_pct*100:.1f}%")

        results[eta] = {
            'cos_globalW_hist': cos_globalW_hist,
            'relerr_globalW_hist': relerr_globalW_hist,
            'norm_ratio_hist': norm_ratio_hist,
            'k_steps_hist': k_steps_hist,
            'cos_steps': cos_steps,
            'avg_steps_per_epoch': avg_steps_per_epoch,
            'layer_wise_metrics': layer_wise_metrics,
            'train_loss_hist': train_loss_hist,
            'test_acc_hist': test_acc_hist,
            'learning_rate_hist': learning_rate_hist,
            'gradient_norm_hist': gradient_norm_hist,
            'gradient_snr': gradient_snr,
            'bias_metrics': bias_metrics,
            'variance_metrics': variance_metrics,
            'early_stop_ratio': early_stop_ratio,
            'convergence_samples': convergence_samples,
        }

    print("\nSaving detailed results to experiment_data_enhanced.pt ...")
    torch.save(results, "experiment_data_seed11_18convergence.pt")
    print("Data Saved.")

    print("Generating Plots...")
    plot_results_icml(results, eta_values)
    plot_convergence_icml(results, eta_values)
    plot_gradient_noise(results, eta_values)
    
if __name__ == "__main__":
    main()