#!/usr/bin/env python3
#Lightweight PnP-DL-ADMM demo (pure NumPy)

#This script implements a tiny Plug-and-Play ADMM solver where the denoiser
#is a dictionary-learning (K-SVD) + OMP patch-based denoiser. It is kept small
#so it runs quickly on modest machines.

#Usage:
#    python3 pnp_dl_admm_light.py

#Dependencies:
#    numpy, matplotlib

#Outputs:
#    Prints PSNR and shows three matplotlib figures (ground truth, measurement, reconstruction).


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, svd

def make_test_image(n=64):
    img = np.zeros((n, n), dtype=np.float32)
    img[8:24, 8:24] = 0.9
    img[35:55, 10:40] = 0.6
    rr, cc = np.ogrid[:n, :n]
    circle = (rr - n*0.7)**2 + (cc - n*0.7)**2 < (n*0.18)**2
    img[circle] = 0.8
    img += (rr / n) * 0.08 + (cc / n) * 0.08
    return np.clip(img, 0, 1)

def gaussian_kernel(size=7, sigma=1.2):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    ker /= ker.sum()
    return ker

def pad_circular(x, pad):
    return np.pad(x, ((pad, pad), (pad, pad)), mode='wrap')

def conv2_circular(x, k):
    p = k.shape[0]//2
    xp = pad_circular(x, p)
    out = np.zeros_like(x)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(xp[i:i+k.shape[0], j:j+k.shape[1]] * k)
    return out

class BlurOperator:
    def __init__(self, kernel):
        self.k = kernel
        self.kt = kernel[::-1, ::-1]
    def A(self, x):
        return conv2_circular(x, self.k)
    def AT(self, x):
        return conv2_circular(x, self.kt)

def extract_patches(img, patch_size=8, stride=8):
    H, W = img.shape
    ph = pw = patch_size
    patches = []
    idxs = []
    for i in range(0, H - ph + 1, stride):
        for j in range(0, W - pw + 1, stride):
            patches.append(img[i:i+ph, j:j+pw].reshape(-1))
            idxs.append((i, j))
    return np.array(patches).T, idxs

def aggregate_patches(patches, idxs, img_shape, patch_size=8, stride=8):
    H, W = img_shape
    ph = pw = patch_size
    out = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)
    k = 0
    for (i, j) in idxs:
        patch = patches[:, k].reshape(ph, pw)
        out[i:i+ph, j:j+pw] += patch
        wgt[i:i+ph, j:j+pw] += 1.0
        k += 1
    wgt[wgt == 0] = 1.0
    return out / wgt

def omp(D, Y, sparsity):
    m, K = D.shape
    _, N = Y.shape
    X = np.zeros((K, N), dtype=np.float32)
    Dt = D.T
    for n in range(N):
        y = Y[:, n]
        residual = y.copy()
        support = []
        for _ in range(sparsity):
            corr = Dt @ residual
            k = np.argmax(np.abs(corr))
            if k in support:
                break
            support.append(k)
            Ds = D[:, support]
            coeffs, *_ = np.linalg.lstsq(Ds, y, rcond=None)
            residual = y - Ds @ coeffs
            if norm(residual) < 1e-6:
                break
        if support:
            X[support, n] = coeffs
    return X

def ksvd(Y, K=64, iters=2, sparsity=4, seed=0):
    rng = np.random.default_rng(seed)
    m, N = Y.shape
    # sample subset if many patches
    if N > 400:
        idx = rng.choice(N, size=400, replace=False)
        Y = Y[:, idx]
        N = Y.shape[1]
    D = rng.normal(size=(m, K)).astype(np.float32)
    D /= np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-8)
    for it in range(iters):
        X = omp(D, Y, sparsity)
        for k in range(K):
            omega = np.nonzero(X[k, :])[0]
            if omega.size == 0:
                d = rng.normal(size=(m,)).astype(np.float32)
                d /= max(norm(d), 1e-8)
                D[:, k] = d
                continue
            Ek = Y[:, omega] - D @ X[:, omega] + np.outer(D[:, k], X[k, omega])
            U, S, Vt = svd(Ek, full_matrices=False)
            D[:, k] = U[:, 0]
            X[k, omega] = S[0] * Vt[0, :]
    return D

def dl_denoise(img, D, patch_size=8, stride=8, sparsity=4):
    Y, idxs = extract_patches(img, patch_size, stride)
    patch_means = Y.mean(axis=0, keepdims=True)
    Yzm = Y - patch_means
    X = omp(D, Yzm, sparsity)
    Y_hat = D @ X + patch_means
    rec = aggregate_patches(Y_hat, idxs, img.shape, patch_size, stride)
    return rec

def pnp_admm(y, Aop, rho=0.6, iters=10, D=None, patch_size=8, stride=8, sparsity=4):
    x = y.copy()
    v = x.copy()
    u = np.zeros_like(x)
    At_y = Aop.AT(y)
    for t in range(iters):
        b = At_y + rho * (v - u)
        x_new = x.copy()
        for _ in range(6):
            Ax = Aop.A(x_new)
            AtAx = Aop.AT(Ax)
            x_new = x_new + (b - (AtAx + rho * x_new)) / (1.1 * (1.0 + rho))
        x = x_new
        v = dl_denoise(x + u, D, patch_size, stride, sparsity)
        u = u + x - v
        print(f"ADMM iter {t+1}/{iters}  residual={norm((x-v).ravel()):.4f}")
    return x

def psnr(x, ref):
    mse = np.mean((x - ref) ** 2)
    if mse == 0:
        return 99.0
    return 10 * np.log10(1.0 / mse)

def main():
    np.random.seed(0)
    img = make_test_image(64)
    ker = gaussian_kernel(size=7, sigma=1.2)
    Aop = BlurOperator(ker)
    sigma_noise = 0.03
    y = Aop.A(img) + sigma_noise * np.random.randn(*img.shape)

    # Train tiny dictionary on measurement patches (fast)
    patch_size = 8
    Ytr, _ = extract_patches(y, patch_size=patch_size, stride=8)
    Ytr = Ytr - Ytr.mean(axis=0, keepdims=True)
    print("Learning tiny dictionary (K-SVD)...")
    D = ksvd(Ytr, K=64, iters=2, sparsity=4, seed=42)

    print("Running PnP-ADMM...")
    x_hat = pnp_admm(y, Aop, rho=0.7, iters=10, D=D, patch_size=patch_size, stride=8, sparsity=4)

    print(f"PSNR(y, img)    = {psnr(y, img):.2f} dB")
    print(f"PSNR(x_hat, img)= {psnr(x_hat, img):.2f} dB")

    plt.figure(figsize=(4,4))
    plt.title(\"Ground truth\")
    plt.imshow(img, cmap=\"gray\")
    plt.axis(\"off\")

    plt.figure(figsize=(4,4))
    plt.title(\"Blurred + Noise (y)\")
    plt.imshow(y, cmap=\"gray\")
    plt.axis(\"off\")
    plt.figure(figsize=(4,4))
    plt.title(\"PnP-ADMM (DL denoiser) Reconstruction\")
    plt.imshow(np.clip(x_hat, 0, 1), cmap=\"gray\")
    plt.axis(\"off\")

    plt.show()

if __name__ == '__main__':
    main()
