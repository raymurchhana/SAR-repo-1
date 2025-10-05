import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve
n
def make_synthetic_satellite_scene(n=256):
    img = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n, n//8):
        img[i:i+n//16, :] = 0.5 + 0.4*(i % 2)
        img[:, i:i+n//16] = 0.5 + 0.4*((i//2) % 2)
    rr = np.arange(n)
    img[rr, (np.round(0.3*rr + 10) % n).astype(int)] = 1.0
    img[(np.round(0.7*rr + 90) % n).astype(int), rr] = 1.0
    for (r, c, s) in [(60, 60, 18), (150, 180, 22), (200, 70, 14)]:
        img[r:r+s, c:c+s] = 0.9
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def gaussian_psf(size=21, sigma=2.0):
    ax = np.arange(-size//2 + 1, size//2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing='xy')
    psf = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    psf /= psf.sum()
    return psf

def grad(x):
    return np.roll(x, -1, axis=1) - x, np.roll(x, -1, axis=0) - x

def div(px, py):
    return (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))

def shrink_iso(px, py, tau):
    mag = np.sqrt(px**2 + py**2 + 1e-12)
    shrink = np.maximum(0, 1 - tau/mag)
    return shrink*px, shrink*py

def deblur_tv_admm(b, k, lam=0.05, rho=1.0, iters=300, tol=1e-4):
    n, m = b.shape
    # FFT precomputation
    K = fft2(np.fft.ifftshift(k), s=b.shape)
    K_conj = np.conj(K)
    KtK = np.abs(K)**2

    wx = 2*np.pi*np.fft.fftfreq(m)
    wy = 2*np.pi*np.fft.fftfreq(n)
    Wx, Wy = np.meshgrid(wx, wy, indexing='xy')
    L = (2 - 2*np.cos(Wx)) + (2 - 2*np.cos(Wy))

    Ktb = (ifft2(K_conj * fft2(b))).real
    inv_denom = 1.0 / (KtK + rho*L + 1e-8)

    # Initialize with Wiener deconvolution (better than b)
    X_init = (K_conj * fft2(b)) / (KtK + 1e-2)
    x = ifft2(X_init).real

    dx = np.zeros_like(b)
    dy = np.zeros_like(b)
    ux = np.zeros_like(b)
    uy = np.zeros_like(b)

    for k_iter in range(iters):
        # x-update
        rhs = Ktb + rho * div(dx - ux, dy - uy)
        X = fft2(rhs) * inv_denom
        x = ifft2(X).real
        x = np.clip(x, 0, 1)   # enforce positivity

        # d-update
        gx, gy = grad(x)
        dx, dy = shrink_iso(gx + ux, gy + uy, lam/rho)

        # u-update
        ux += gx - dx
        uy += gy - dy

        # Primal & dual residual check
        r_norm = np.sqrt(np.sum((gx - dx)**2 + (gy - dy)**2))
        s_norm = rho * np.sqrt(np.sum((dx - np.roll(dx, 1, axis=1))**2 +
                                      (dy - np.roll(dy, 1, axis=0))**2))
        if r_norm < tol and s_norm < tol:
            print(f"Converged at iteration {k_iter}")
            break

    return np.clip(x, 0, 1)


# Demo
np.random.seed(0)
x_true = make_synthetic_satellite_scene(256)
psf = gaussian_psf(size=21, sigma=2.0)
blurred = fftconvolve(x_true, psf, mode='same')
sigma_noise = 0.02
b = np.clip(blurred + sigma_noise*np.random.randn(*blurred.shape), 0, 1)

x_rec = deblur_tv_admm(b, psf, lam=0.02, rho=2.0, iters=300)

# Display
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(x_true, cmap='gray', vmin=0, vmax=1)
axs[0].set_title("Ground Truth"); axs[0].axis('off')
axs[1].imshow(b, cmap='gray', vmin=0, vmax=1)
axs[1].set_title("Blurred + noisy"); axs[1].axis('off')
axs[2].imshow(x_rec, cmap='gray', vmin=0, vmax=1)
axs[2].set_title("ADMM-TV (optimized)"); axs[2].axis('off')
plt.tight_layout(); plt.show()

psnr = lambda x, y: 10*np.log10(1.0 / (np.mean((x-y)**2) + 1e-12))
print("PSNR(observation vs truth):", psnr(b, x_true))
print("PSNR(recon vs truth):      ", psnr(x_rec, x_true))
