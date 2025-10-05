# Re-run after state reset

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve

def make_synthetic_satellite_scene(n=256):
    img = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n, n//8):
        img[i:i+n//16, :] = 0.5 + 0.4*(i % 2)
        img[:, i:i+n//16] = 0.5 + 0.4*((i//2) % 2)
    rr = np.linspace(0, n-1, n)
    img[np.arange(n), (np.round(0.3*rr + 10) % n).astype(int)] = 1.0
    img[(np.round(0.7*rr + 90) % n).astype(int), np.arange(n)] = 1.0
    for (r, c, s) in [(60, 60, 18), (150, 180, 22), (200, 70, 14)]:
        img[r:r+s, c:c+s] = 0.9
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img

#generating a 2D Gaussian blur kernel (Point Spread Function).
#It is later used to blur the synthetic image to
#simulate how a satellite sensor blurs details.
def gaussian_psf(size=21, sigma=2.0):
    ax = np.arange(-size//2 + 1, size//2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing='xy')
    psf = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    psf /= psf.sum()
    return psf

def grad(x):
    gx = np.roll(x, -1, axis=1) - x
    gy = np.roll(x, -1, axis=0) - x
    return gx, gy

def div(px, py):
    cx = px - np.roll(px, 1, axis=1)
    cy = py - np.roll(py, 1, axis=0)
    return cx + cy

def tv_iso(px, py, eps=1e-12):
    return np.sqrt(px*px + py*py + eps)

def shrink_iso(px, py, tau):
    mag = tv_iso(px, py)
    denom = np.maximum(1.0, mag / np.maximum(tau, 1e-12))
    return px / denom, py / denom

def deblur_tv_admm(b, k, lam=0.02, rho=1.0, iters=100):
    n, m = b.shape
    # PSF must be ifftshifted so its center matches FFT indexing
    K = fft2(np.fft.ifftshift(k), s=b.shape)
    K_conj = np.conj(K)
    KtK = np.abs(K)**2

    wx = 2*np.pi*np.fft.fftfreq(m)
    wy = 2*np.pi*np.fft.fftfreq(n)
    Wx, Wy = np.meshgrid(wx, wy, indexing='xy')
    L = (2 - 2*np.cos(Wx)) + (2 - 2*np.cos(Wy))

    x = b.copy()
    dx = np.zeros_like(b)
    dy = np.zeros_like(b)
    ux = np.zeros_like(b)
    uy = np.zeros_like(b)
    Ktb = np.real(ifft2(K_conj * fft2(b)))

    for k_iter in range(iters):
        vx = dx - ux
        vy = dy - uy
        rhs = Ktb + rho * div(vx, vy)
        X = fft2(rhs) / (KtK + rho * L + 1e-8)
        x = np.real(ifft2(X))

        # stability check: stop early if values blow up or NaNs/Inf appear
        if not np.isfinite(x).all() or np.nanmax(np.abs(x)) > 1e8:
            print(f"deblur_tv_admm: stopping early at iter {k_iter} due to instability "
                  f"(min={np.nanmin(x)}, max={np.nanmax(x)})")
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            break

        gx, gy = grad(x)
        qx = gx + ux
        qy = gy + uy
        dx, dy = shrink_iso(qx, qy, lam / rho)
        ux = ux + gx - dx
        uy = uy + gy - dy

    return x

# Demo
np.random.seed(0)
x_true = make_synthetic_satellite_scene(256)
psf = gaussian_psf(size=21, sigma=2.0)
blurred = fftconvolve(x_true, psf, mode='same')
sigma_noise = 0.02
b = np.clip(blurred + sigma_noise*np.random.randn(*blurred.shape), 0, 1)

x_rec = deblur_tv_admm(b, psf, lam=0.03, rho=1.2, iters=120)

# Debug / sanitize before plotting to avoid matplotlib overflow on cursor
print("recon stats before sanitize:",
      "min=", np.nanmin(x_rec), "max=", np.nanmax(x_rec),
      "has_nan=", np.isnan(x_rec).any(), "has_inf=", np.isinf(x_rec).any())

x_rec = np.nan_to_num(x_rec, nan=0.0, posinf=1.0, neginf=0.0)
# optionally scale/clamp for display
x_rec = np.clip(x_rec, 0.0, 1.0)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(x_true, cmap='gray', vmin=0, vmax=1)
axs[0].set_title("Synthetic 'satellite' ground truth")
axs[0].axis('off')
axs[1].imshow(b, cmap='gray', vmin=0, vmax=1)
axs[1].set_title("Blurred + noisy observation")
axs[1].axis('off')
axs[2].imshow(x_rec, cmap='gray', vmin=0, vmax=1)
axs[2].set_title("ADMM-TV reconstruction")
axs[2].axis('off')
plt.tight_layout()
plt.show()

psnr = lambda x, y: 10*np.log10(1.0**2 / (np.mean((x-y)**2) + 1e-12))
print("PSNR(observation vs truth):", psnr(b, x_true))
print("PSNR(recon vs truth):      ", psnr(x_rec, x_true))
