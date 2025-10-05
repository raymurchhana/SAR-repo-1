#creating a synthetic satellite image with geometric patterns

import numpy as np
import matplotlib.pyplot as plt

def make_synthetic_satellite_scene(n=256):
    img = np.zeros((n, n), dtype=np.float32)

    # Add some grid-like patterns
    for i in range(0, n, n//8):
        img[i:i+n//16, :] = 0.5 + 0.4*(i % 2)
        img[:, i:i+n//16] = 0.5 + 0.4*((i//2) % 2)

    # Add diagonal lines
    rr = np.linspace(0, n-1, n)
    img[np.arange(n), (np.round(0.3*rr + 10) % n).astype(int)] = 1.0
    img[(np.round(0.7*rr + 90) % n).astype(int), np.arange(n)] = 1.0

    # Add some square patches
    for (r, c, s) in [(60, 60, 18), (150, 180, 22), (200, 70, 14)]:
        img[r:r+s, c:c+s] = 0.9

    # Normalize to [0,1]
    img = img - img.min()
    img = img / (img.max() + 1e-8)

    return img

# Run the function
image = make_synthetic_satellite_scene(256)

# Display the image
plt.imshow(image, cmap="gray")
plt.title("Synthetic Satellite Scene")
plt.axis("off")
plt.show()
