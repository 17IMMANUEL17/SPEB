import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
from torch.nn.grad import conv2d_weight
import numpy as np

torch.set_default_dtype(torch.float32)

# -----------------------------
# Activations & Helpers
# -----------------------------
def relu(u): return torch.relu(u)
def relu_prime(u): return (u > 0).to(u.dtype)

def flat_cat(tup):
    return torch.cat([t.reshape(-1) for t in tup], dim=0)

def cos_sim(a, b, eps=1e-12):
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b) / denom)

def relative_error(a, b, eps=1e-12):
    return float((a - b).norm() / b.norm().clamp_min(eps))

# -----------------------------
# Data Augmentation: Cutout
# -----------------------------
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img is a Tensor here (C, H, W)
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask).to(img.device)
        mask = mask.expand_as(img)
        return img * mask

# -----------------------------
# Model: VGG-11 Style (8 Conv + 1 FC for CIFAR)
# -----------------------------
@dataclass
class CNN9:
    # Block 1
    W1: torch.Tensor; b1: torch.Tensor   # 3->64
    W2: torch.Tensor; b2: torch.Tensor   # 64->128 (Pool)
    # Block 2
    W3: torch.Tensor; b3: torch.Tensor   # 128->256
    W4: torch.Tensor; b4: torch.Tensor   # 256->256 (Pool)
    # Block 3
    W5: torch.Tensor; b5: torch.Tensor   # 256->512
    W6: torch.Tensor; b6: torch.Tensor   # 512->512 (Pool)
    # Block 4
    W7: torch.Tensor; b7: torch.Tensor   # 512->512
    W8: torch.Tensor; b8: torch.Tensor   # 512->512 (Pool)
    # Classifier
    W9: torch.Tensor; b9: torch.Tensor   # 512*2*2 -> 10

    @property
    def device(self): return self.W1.device

# -----------------------------
# Forward (Expanded for 9 Layers)
# -----------------------------
@torch.no_grad()
def forward_u_sig(net: CNN9, x0, m1, m2, m3, m4, m5, m6, m7, m8):
    # L1: 3->64
    u1 = F.conv2d(x0, net.W1, net.b1, stride=1, padding=1)
    sig1 = relu(u1)
    # L2: 64->128 (Stride 2)
    u2 = F.conv2d(m1, net.W2, net.b2, stride=2, padding=1)
    sig2 = relu(u2)
    # L3: 128->256
    u3 = F.conv2d(m2, net.W3, net.b3, stride=1, padding=1)
    sig3 = relu(u3)
    # L4: 256->256 (Stride 2)
    u4 = F.conv2d(m3, net.W4, net.b4, stride=2, padding=1)
    sig4 = relu(u4)
    # L5: 256->512
    u5 = F.conv2d(m4, net.W5, net.b5, stride=1, padding=1)
    sig5 = relu(u5)
    # L6: 512->512 (Stride 2)
    u6 = F.conv2d(m5, net.W6, net.b6, stride=2, padding=1)
    sig6 = relu(u6)
    # L7: 512->512
    u7 = F.conv2d(m6, net.W7, net.b7, stride=1, padding=1)
    sig7 = relu(u7)
    # L8: 512->512 (Stride 2)
    u8 = F.conv2d(m7, net.W8, net.b8, stride=2, padding=1)
    sig8 = relu(u8)
    
    # L9: Linear
    B = x0.shape[0]
    m8_flat = m8.reshape(B, -1)
    u9 = m8_flat @ net.W9.t() + net.b9
    sig9 = u9

    return (u1, sig1), (u2, sig2), (u3, sig3), (u4, sig4), \
           (u5, sig5), (u6, sig6), (u7, sig7), (u8, sig8), (u9, sig9)

class XZState:
    def __init__(self):
        self.x1 = None; self.z1 = None
        self.x2 = None; self.z2 = None
        self.x3 = None; self.z3 = None
        self.x4 = None; self.z4 = None
        self.x5 = None; self.z5 = None
        self.x6 = None; self.z6 = None
        self.x7 = None; self.z7 = None
        self.x8 = None; self.z8 = None
        self.x9 = None; self.z9 = None

# -----------------------------
# Hamiltonian Relaxation Gradient (Fixed Scope & Stability)
# -----------------------------
@torch.no_grad()
def xz_relax_batch_grad(
    net: CNN9, x0, y,
    eta=1,  # REDUCED from 1.0 for Stability
    K=60,     # Increased steps slightly
    state: XZState | None = None,
    tol: float = 1e-6,
    warm_start: bool = True,
    beta: float = 1.0,
):
    device = net.device
    B = x0.shape[0]
    y_onehot = F.one_hot(y, num_classes=10).to(x0.dtype)

    # 1. Allocation Function
    def alloc():
        x1 = torch.zeros(B, 64, 32, 32, device=device);   z1 = torch.zeros_like(x1)
        x2 = torch.zeros(B, 128, 16, 16, device=device);  z2 = torch.zeros_like(x2)
        x3 = torch.zeros(B, 256, 16, 16, device=device);  z3 = torch.zeros_like(x3)
        x4 = torch.zeros(B, 256, 8, 8, device=device);    z4 = torch.zeros_like(x4)
        x5 = torch.zeros(B, 512, 8, 8, device=device);    z5 = torch.zeros_like(x5)
        x6 = torch.zeros(B, 512, 4, 4, device=device);    z6 = torch.zeros_like(x6)
        x7 = torch.zeros(B, 512, 4, 4, device=device);    z7 = torch.zeros_like(x7)
        x8 = torch.zeros(B, 512, 2, 2, device=device);    z8 = torch.zeros_like(x8)
        x9 = torch.zeros(B, 10, device=device);           z9 = torch.zeros_like(x9)
        return x1,z1,x2,z2,x3,z3,x4,z4,x5,z5,x6,z6,x7,z7,x8,z8,x9,z9

    # 2. Variable Initialization (Fixing UnboundLocalError)
    if (state is None) or (not warm_start) or (state.x1 is None) or (state.x1.shape[0] != B):
        x1,z1,x2,z2,x3,z3,x4,z4,x5,z5,x6,z6,x7,z7,x8,z8,x9,z9 = alloc()
        if state is not None:
            (state.x1,state.z1,state.x2,state.z2,state.x3,state.z3,state.x4,state.z4,state.x5,state.z5,state.x6,state.z6,state.x7,state.z7,state.x8,state.z8,state.x9,state.z9) = \
            (x1,z1,x2,z2,x3,z3,x4,z4,x5,z5,x6,z6,x7,z7,x8,z8,x9,z9)
    else:
        x1,z1,x2,z2,x3,z3,x4,z4,x5,z5,x6,z6,x7,z7,x8,z8,x9,z9 = \
        (state.x1,state.z1,state.x2,state.z2,state.x3,state.z3,state.x4,state.z4,state.x5,state.z5,state.x6,state.z6,state.x7,state.z7,state.x8,state.z8,state.x9,state.z9)

    # 3. Relaxation Loop
    steps_taken = 0
    for _ in range(K):
        steps_taken += 1
        m1 = (x1 + z1) * 0.5; s1 = (x1 - z1)
        m2 = (x2 + z2) * 0.5; s2 = (x2 - z2)
        m3 = (x3 + z3) * 0.5; s3 = (x3 - z3)
        m4 = (x4 + z4) * 0.5; s4 = (x4 - z4)
        m5 = (x5 + z5) * 0.5; s5 = (x5 - z5)
        m6 = (x6 + z6) * 0.5; s6 = (x6 - z6)
        m7 = (x7 + z7) * 0.5; s7 = (x7 - z7)
        m8 = (x8 + z8) * 0.5; s8 = (x8 - z8)
        m9 = (x9 + z9) * 0.5; s9 = (x9 - z9)

        (u1, sig1), (u2, sig2), (u3, sig3), (u4, sig4), (u5, sig5), (u6, sig6), (u7, sig7), (u8, sig8), (u9, sig9) = \
            forward_u_sig(net, x0, m1, m2, m3, m4, m5, m6, m7, m8)

        # Residuals
        F1 = sig1 - m1; F2 = sig2 - m2; F3 = sig3 - m3
        F4 = sig4 - m4; F5 = sig5 - m5; F6 = sig6 - m6
        F7 = sig7 - m7; F8 = sig8 - m8; F9 = sig9 - m9

        # Target forcing
        p = torch.softmax(m9, dim=1)
        g9 = (p - y_onehot)

        # Adjoints
        q2 = relu_prime(u2) * s2
        q3 = relu_prime(u3) * s3
        q4 = relu_prime(u4) * s4
        q5 = relu_prime(u5) * s5
        q6 = relu_prime(u6) * s6
        q7 = relu_prime(u7) * s7
        q8 = relu_prime(u8) * s8
        q9 = s9

        # Transposed Convs
        WTq8 = (q9 @ net.W9).reshape(B, 512, 2, 2)
        WTq7 = F.conv_transpose2d(q8, net.W8, bias=None, stride=2, padding=1, output_padding=1)
        WTq6 = F.conv_transpose2d(q7, net.W7, bias=None, stride=1, padding=1)
        WTq5 = F.conv_transpose2d(q6, net.W6, bias=None, stride=2, padding=1, output_padding=1)
        WTq4 = F.conv_transpose2d(q5, net.W5, bias=None, stride=1, padding=1)
        WTq3 = F.conv_transpose2d(q4, net.W4, bias=None, stride=2, padding=1, output_padding=1)
        WTq2 = F.conv_transpose2d(q3, net.W3, bias=None, stride=1, padding=1)
        WTq1 = F.conv_transpose2d(q2, net.W2, bias=None, stride=2, padding=1, output_padding=1)

        Jt1 = -s1 + WTq1; Jt2 = -s2 + WTq2; Jt3 = -s3 + WTq3
        Jt4 = -s4 + WTq4; Jt5 = -s5 + WTq5; Jt6 = -s6 + WTq6
        Jt7 = -s7 + WTq7; Jt8 = -s8 + WTq8; Jt9 = -s9

        # Updates
        dx1 = F1 + 0.5*Jt1; dz1 = F1 - 0.5*Jt1
        dx2 = F2 + 0.5*Jt2; dz2 = F2 - 0.5*Jt2
        dx3 = F3 + 0.5*Jt3; dz3 = F3 - 0.5*Jt3
        dx4 = F4 + 0.5*Jt4; dz4 = F4 - 0.5*Jt4
        dx5 = F5 + 0.5*Jt5; dz5 = F5 - 0.5*Jt5
        dx6 = F6 + 0.5*Jt6; dz6 = F6 - 0.5*Jt6
        dx7 = F7 + 0.5*Jt7; dz7 = F7 - 0.5*Jt7
        dx8 = F8 + 0.5*Jt8; dz8 = F8 - 0.5*Jt8
        
        dx9 = F9 + 0.5*Jt9 + 0.5*beta*g9
        dz9 = F9 - 0.5*Jt9 - 0.5*beta*g9

        x1.add_(dx1, alpha=eta); z1.add_(dz1, alpha=eta)
        x2.add_(dx2, alpha=eta); z2.add_(dz2, alpha=eta)
        x3.add_(dx3, alpha=eta); z3.add_(dz3, alpha=eta)
        x4.add_(dx4, alpha=eta); z4.add_(dz4, alpha=eta)
        x5.add_(dx5, alpha=eta); z5.add_(dz5, alpha=eta)
        x6.add_(dx6, alpha=eta); z6.add_(dz6, alpha=eta)
        x7.add_(dx7, alpha=eta); z7.add_(dz7, alpha=eta)
        x8.add_(dx8, alpha=eta); z8.add_(dz8, alpha=eta)
        x9.add_(dx9, alpha=eta); z9.add_(dz9, alpha=eta)

        upd = (dx1.abs().mean() + dx5.abs().mean() + dx9.abs().mean()).item()
        if upd < tol: break

    # 4. Compute Gradients from State
    m1 = (x1 + z1) * 0.5; s1 = (x1 - z1)
    m2 = (x2 + z2) * 0.5; s2 = (x2 - z2)
    m3 = (x3 + z3) * 0.5; s3 = (x3 - z3)
    m4 = (x4 + z4) * 0.5; s4 = (x4 - z4)
    m5 = (x5 + z5) * 0.5; s5 = (x5 - z5)
    m6 = (x6 + z6) * 0.5; s6 = (x6 - z6)
    m7 = (x7 + z7) * 0.5; s7 = (x7 - z7)
    m8 = (x8 + z8) * 0.5; s8 = (x8 - z8)

    (u1, _), (u2, _), (u3, _), (u4, _), (u5, _), (u6, _), (u7, _), (u8, _), (_, _) = \
        forward_u_sig(net, x0, m1, m2, m3, m4, m5, m6, m7, m8)

    delta1 = relu_prime(u1) * s1; delta2 = relu_prime(u2) * s2
    delta3 = relu_prime(u3) * s3; delta4 = relu_prime(u4) * s4
    delta5 = relu_prime(u5) * s5; delta6 = relu_prime(u6) * s6
    delta7 = relu_prime(u7) * s7; delta8 = relu_prime(u8) * s8
    delta9 = s9

    dW1 = conv2d_weight(x0, net.W1.shape, delta1, stride=1, padding=1) / B
    dW2 = conv2d_weight(m1, net.W2.shape, delta2, stride=2, padding=1) / B
    dW3 = conv2d_weight(m2, net.W3.shape, delta3, stride=1, padding=1) / B
    dW4 = conv2d_weight(m3, net.W4.shape, delta4, stride=2, padding=1) / B
    dW5 = conv2d_weight(m4, net.W5.shape, delta5, stride=1, padding=1) / B
    dW6 = conv2d_weight(m5, net.W6.shape, delta6, stride=2, padding=1) / B
    dW7 = conv2d_weight(m6, net.W7.shape, delta7, stride=1, padding=1) / B
    dW8 = conv2d_weight(m7, net.W8.shape, delta8, stride=2, padding=1) / B
    
    m8_flat = m8.reshape(B, -1)
    dW9 = (delta9.t() @ m8_flat) / B

    db1 = delta1.sum(dim=(0,2,3)) / B; db2 = delta2.sum(dim=(0,2,3)) / B
    db3 = delta3.sum(dim=(0,2,3)) / B; db4 = delta4.sum(dim=(0,2,3)) / B
    db5 = delta5.sum(dim=(0,2,3)) / B; db6 = delta6.sum(dim=(0,2,3)) / B
    db7 = delta7.sum(dim=(0,2,3)) / B; db8 = delta8.sum(dim=(0,2,3)) / B
    db9 = delta9.mean(dim=0)

    ce = F.cross_entropy(m9, y).item()
    return (dW1,dW2,dW3,dW4,dW5,dW6,dW7,dW8,dW9), \
           (db1,db2,db3,db4,db5,db6,db7,db8,db9), ce, steps_taken

# -----------------------------
# SGD + Clip
# -----------------------------
@torch.no_grad()
def sgd_momentum_step(net: CNN9, gradsW, gradsb, vW, vb,
                      lr=0.01, momentum=0.9, weight_decay=5e-4, clip=1.0):
    for i in range(9):
        dWi = gradsW[i] + weight_decay * getattr(net, f"W{i+1}")
        dbi = gradsb[i]
        
        # Gradient Clipping (Essential for stability)
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
        
        h = relu(F.conv2d(x, net.W1, net.b1, stride=1, padding=1))
        h = relu(F.conv2d(h, net.W2, net.b2, stride=2, padding=1))
        h = relu(F.conv2d(h, net.W3, net.b3, stride=1, padding=1))
        h = relu(F.conv2d(h, net.W4, net.b4, stride=2, padding=1))
        h = relu(F.conv2d(h, net.W5, net.b5, stride=1, padding=1))
        h = relu(F.conv2d(h, net.W6, net.b6, stride=2, padding=1))
        h = relu(F.conv2d(h, net.W7, net.b7, stride=1, padding=1))
        h = relu(F.conv2d(h, net.W8, net.b8, stride=2, padding=1))
        
        logits = h.reshape(x.size(0), -1) @ net.W9.t() + net.b9
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def autograd_grads_like_cnn9(net: CNN9, x, y):
    params = {}
    for i in range(1, 10):
        params[f"W{i}"] = getattr(net, f"W{i}").detach().clone().requires_grad_(True)
        params[f"b{i}"] = getattr(net, f"b{i}").detach().clone().requires_grad_(True)
        
    h = relu(F.conv2d(x, params['W1'], params['b1'], stride=1, padding=1))
    h = relu(F.conv2d(h, params['W2'], params['b2'], stride=2, padding=1))
    h = relu(F.conv2d(h, params['W3'], params['b3'], stride=1, padding=1))
    h = relu(F.conv2d(h, params['W4'], params['b4'], stride=2, padding=1))
    h = relu(F.conv2d(h, params['W5'], params['b5'], stride=1, padding=1))
    h = relu(F.conv2d(h, params['W6'], params['b6'], stride=2, padding=1))
    h = relu(F.conv2d(h, params['W7'], params['b7'], stride=1, padding=1))
    h = relu(F.conv2d(h, params['W8'], params['b8'], stride=2, padding=1))

    logits = h.reshape(x.size(0), -1) @ params['W9'].t() + params['b9']
    
    loss = F.cross_entropy(logits, y)
    loss.backward()
    return tuple(params[f"W{i}"].grad for i in range(1, 10)), tuple(params[f"b{i}"].grad for i in range(1, 10)), float(loss.detach())

# -----------------------------
# Initialization (Scaled for Stability)
# -----------------------------
def kaiming_init_scaled(shape, device, scale_factor=1.0):
    if len(shape) == 4: fan_in = shape[1] * shape[2] * shape[3]
    else: fan_in = shape[1]
    std = math.sqrt(2.0 / fan_in) * scale_factor
    return std * torch.randn(shape, device=device)

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cifar_mean = (0.4914, 0.4822, 0.4465); cifar_std  = (0.2470, 0.2435, 0.2616)

    # Pipeline
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
        Cutout(n_holes=1, length=8)
    ])
    test_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)])

    train_loader = DataLoader(datasets.CIFAR10("./data", train=True, download=True, transform=train_tfm),
                              batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(datasets.CIFAR10("./data", train=False, download=True, transform=test_tfm),
                              batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    epochs = 100; K = 80; eta_values = [1.0] 
    results = {}

    for eta in eta_values:
        print(f"\n{'='*40}\nTraining VGG-11 with eta={eta}\n{'='*40}")
        torch.manual_seed(42)
        
        scale = 0.8
        W1 = kaiming_init_scaled((64, 3, 3, 3), device, scale)
        W2 = kaiming_init_scaled((128, 64, 3, 3), device, scale)
        W3 = kaiming_init_scaled((256, 128, 3, 3), device, scale)
        W4 = kaiming_init_scaled((256, 256, 3, 3), device, scale)
        W5 = kaiming_init_scaled((512, 256, 3, 3), device, scale)
        W6 = kaiming_init_scaled((512, 512, 3, 3), device, scale)
        W7 = kaiming_init_scaled((512, 512, 3, 3), device, scale)
        W8 = kaiming_init_scaled((512, 512, 3, 3), device, scale)
        W9 = kaiming_init_scaled((10, 512*2*2), device, scale)

        b1 = torch.zeros(64, device=device); b2 = torch.zeros(128, device=device)
        b3 = torch.zeros(256, device=device); b4 = torch.zeros(256, device=device)
        b5 = torch.zeros(512, device=device); b6 = torch.zeros(512, device=device)
        b7 = torch.zeros(512, device=device); b8 = torch.zeros(512, device=device)
        b9 = torch.zeros(10, device=device)

        net = CNN9(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,W9,b9)
        ema_net = CNN9(W1.clone(),b1.clone(),W2.clone(),b2.clone(),W3.clone(),b3.clone(),W4.clone(),b4.clone(),W5.clone(),b5.clone(),W6.clone(),b6.clone(),W7.clone(),b7.clone(),W8.clone(),b8.clone(),W9.clone(),b9.clone())

        vW = [torch.zeros_like(getattr(net, f"W{i}")) for i in range(1,10)]
        vb = [torch.zeros_like(getattr(net, f"b{i}")) for i in range(1,10)]

        global_step = 0; state = XZState()
        cos_hist = []; err_hist = []; steps_hist = []

        for ep in range(1, epochs+1):
            running_ce = 0.0
            running_steps = 0.0 # <--- NEW: Initialize step counter
            
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                lr = 0.05 * (0.5 * (1 + math.cos(math.pi * global_step / (epochs * len(train_loader))))) + 1e-4

                # <--- CHANGED: Capture 'steps' from return values (was previously '_')
                gradsW, gradsb, ce, steps = xz_relax_batch_grad(net, x, y, eta=eta, K=K, state=state, warm_start=True)

                if global_step % 200 == 0:
                    gW_ag, _, _ = autograd_grads_like_cnn9(net, x, y)
                    sim = cos_sim(flat_cat(gradsW), flat_cat(gW_ag))
                    err = relative_error(flat_cat(gradsW), flat_cat(gW_ag))
                    cos_hist.append(sim); err_hist.append(err); steps_hist.append(global_step)
                    # <--- UPDATED: Added steps to the periodic log as well
                    print(f"[Step {global_step}] CosSim: {sim:.4f} | RelErr: {err:.4f} | Loss: {ce:.3f} | Steps: {steps}")

                sgd_momentum_step(net, gradsW, gradsb, vW, vb, lr=lr, clip=1.0)
                ema_update(ema_net, net)
                running_ce += ce
                running_steps += steps # <--- NEW: Accumulate steps
                global_step += 1

            test_acc = accuracy(ema_net, test_loader, device)
            
            # <--- UPDATED: Calculate average and add to print string
            avg_steps = running_steps / len(train_loader)
            print(f"Epoch {ep} | Loss: {running_ce/len(train_loader):.3f} | Avg Steps: {avg_steps:.1f} | Test Acc: {test_acc*100:.2f}%")

        results[eta] = {'cos_globalW_hist': cos_hist, 'relerr_globalW_hist': err_hist, 'cos_steps': steps_hist}
        plot_results_icml(results, [eta])
        
    print(f"Final Accuracy: {accuracy(ema_net, test_loader, device, 2000)*100:.2f}%")

def plot_results_icml(results, eta_values):
    plt.figure(figsize=(8,5))
    for eta in eta_values:
        if eta in results:
            plt.plot(results[eta]['cos_steps'], results[eta]['relerr_globalW_hist'], label=f'eta={eta}')
    plt.yscale('log'); plt.legend(); plt.title('Gradient Fidelity (VGG-11)'); plt.show()

if __name__ == "__main__":
    main()