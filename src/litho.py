import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft2, ifft2

# Grid parameters
grid_size_x = 140 * 3
grid_size_y = 400 * 3

# Create photomask pattern
def create_mask(grid_size_x, grid_size_y):
    mask = np.ones((grid_size_x, grid_size_y))
    for ii in [0, 2, 4]:
        c = 80 * ii
        mask[0 * 3:80 * 3 + 40, (0 + c) * 3:(10 + c) * 3] = 0
        mask[90 * 3 + 40:140 * 3 + 40, (0 + c) * 3:(10 + c) * 3] = 0
        mask[70 * 3 + 40:80 * 3 + 40, (10 + c) * 3:(20 + c) * 3] = 0
        mask[0 * 3:60 * 3 + 40, (20 + c) * 3:(30 + c) * 3] = 0
        mask[70 * 3 + 40:140 * 3 + 40, (20 + c) * 3:(30 + c) * 3] = 0
        mask[50 * 3 + 40:60 * 3 + 40, (30 + c) * 3:(40 + c) * 3] = 0
        mask[0 * 3:40 * 3 + 40, (40 + c) * 3:(50 + c) * 3] = 0
        mask[50 * 3 + 40:140 * 3 + 40, (40 + c) * 3:(50 + c) * 3] = 0
        mask[30 * 3 + 40:40 * 3 + 40, (50 + c) * 3:(60 + c) * 3] = 0
        mask[0 * 3:40 * 3 - 20, (60 + c) * 3:(70 + c) * 3] = 0
        mask[30 * 3 + 40:140 * 3 + 40, (60 + c) * 3:(70 + c) * 3] = 0
    for ii in [1, 3]:
        c = 80 * ii
        mask[0 * 3:80 * 3 - 40, (0 + c) * 3:(10 + c) * 3] = 0
        mask[90 * 3 - 40:140 * 3 + 20, (0 + c) * 3:(10 + c) * 3] = 0
        mask[70 * 3 - 40:80 * 3 - 40, (10 + c) * 3:(20 + c) * 3] = 0
        mask[0 * 3:60 * 3 - 40, (20 + c) * 3:(30 + c) * 3] = 0
        mask[70 * 3 - 40:140 * 3 + 20, (20 + c) * 3:(30 + c) * 3] = 0
        mask[50 * 3 - 40:60 * 3 - 40, (30 + c) * 3:(40 + c) * 3] = 0
        mask[0 * 3:40 * 3 - 40, (40 + c) * 3:(50 + c) * 3] = 0
        mask[50 * 3 - 40:140 * 3 + 20, (40 + c) * 3:(50 + c) * 3] = 0
        mask[30 * 3 - 40:40 * 3 - 40, (50 + c) * 3:(60 + c) * 3] = 0
        mask[0 * 3:20 * 3 - 40, (60 + c) * 3:(70 + c) * 3] = 0
        mask[30 * 3 - 40:140 * 3 + 20, (60 + c) * 3:(70 + c) * 3] = 0
    return mask

# Create quasar-shaped light source
def create_light_source(grid_size_x, grid_size_y, inner_radius, outer_radius, critical_angle, sigma):
    aperture_size = min([grid_size_x, grid_size_y])
    x = np.linspace(-1, 1, aperture_size)
    y = np.linspace(-1, 1, aperture_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X) + np.pi
    mask = (R >= inner_radius) & (R <= outer_radius) & (
        (Theta <= critical_angle / 2) |
        ((Theta >= 0.5 * np.pi - critical_angle / 2) & (Theta <= 0.5 * np.pi + critical_angle / 2)) |
        ((Theta >= np.pi - critical_angle / 2) & (Theta <= np.pi + critical_angle / 2)) |
        ((Theta >= 1.5 * np.pi - critical_angle / 2) & (Theta <= 1.5 * np.pi + critical_angle / 2)) |
        (Theta >= 2 * np.pi - critical_angle / 2)
    )
    aperture = np.zeros((aperture_size, aperture_size))
    aperture[mask] = np.exp(-((R[mask] - inner_radius) ** 2) / (2 * sigma ** 2))
    aperture /= aperture.max()
    source = np.zeros((grid_size_x, grid_size_y))
    start_x = (grid_size_y - aperture_size) // 2
    start_y = (grid_size_x - aperture_size) // 2
    source[start_y:start_y + aperture_size, start_x:start_x + aperture_size] = aperture
    return source

# Perform Fourier optics simulation
def optical_simulation(mask, source):
    mask_ft = fftshift(fft2(mask))
    source_ft = fftshift(fft2(source))
    result_ft = mask_ft * source_ft
    result = np.abs(ifft2(result_ft)) ** 2
    return result / result.max()

# Parameters
inner_radius = 0.6
outer_radius = 0.8
critical_angle = 44 * np.pi / 180
sigma = 1

# Generate pattern and simulate
mask = create_mask(grid_size_x, grid_size_y)
light_source = create_light_source(grid_size_x, grid_size_y, inner_radius, outer_radius, critical_angle, sigma)
result = optical_simulation(mask, light_source)

# Display standard deviation
std_dev = np.std(result)
print("Standard Deviation:", std_dev)

# Plot mask and light source
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Mask Pattern")
plt.imshow(mask, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Light Source Intensity")
plt.imshow(light_source, cmap='hot')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.show()

# 3D plot of result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(result.shape[1]), np.arange(result.shape[0]))
surf = ax.plot_surface(X, Y, result, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Normalized Intensity")
ax.set_title("3D Light Intensity Distribution")
ax.set_xlabel("X-axis (pixels)")
ax.set_ylabel("Y-axis (pixels)")
ax.set_zlabel("Normalized Intensity")
ax.view_init(elev=30, azim=120)
plt.show()
