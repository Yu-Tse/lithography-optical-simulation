import os
import numpy as np
import pandas as pd
from scipy.fftpack import fftshift, fft2, ifft2
from tqdm import tqdm

# Parameters
grid_size_x = 140 * 3
grid_size_y = 400 * 3

# Create mask
def create_mask(grid_size_x, grid_size_y):
    mask = np.ones((grid_size_x, grid_size_y))
    for ii in [0,2,4]:
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
    for ii in [1,3]:
        c = 80 * ii
        mask[0 * 3:80 * 3 - 40, (0 + c) * 3:(10 + c) * 3] = 0
        mask[90 * 3 - 40:140 * 3 + 20, (0 + c) * 3:(10 + c) * 3] = 0
        mask[70 * 3 - 40:80 * 3 - 40, (10 + c) * 3:(20 + c) * 3] = 0
        mask[0 * 3:60 * 3 - 40, (20 + c) * 3:(30 + c) * 3] = 0
        mask[70 * 3- 40:140 * 3+ 20, (20 + c) * 3:(30 + c) * 3] = 0
        mask[50 * 3- 40:60 * 3- 40, (30 + c) * 3:(40 + c) * 3] = 0
        mask[0 * 3:40 * 3- 40, (40 + c) * 3:(50 + c) * 3] = 0
        mask[50 * 3- 40:140 * 3+ 20, (40 + c) * 3:(50 + c) * 3] = 0
        mask[30 * 3- 40:40 * 3- 40, (50 + c) * 3:(60 + c) * 3] = 0
        mask[0 * 3:20 * 3- 40, (60 + c) * 3:(70 + c) * 3] = 0
        mask[30 * 3- 40:140 * 3+ 20, (60 + c) * 3:(70 + c) * 3] = 0
    return mask

# Create light source
def create_light_source(grid_size_x, grid_size_y, inner_radius, outer_radius, critical_angle, sigma):
    source = np.zeros((grid_size_x, grid_size_y))
    aperture_size = min([grid_size_x, grid_size_y])
    x = np.linspace(-1, 1, aperture_size)
    y = np.linspace(-1, 1, aperture_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X) + np.pi
    Theta_rotated = (Theta - 0.25 * np.pi) % (2 * np.pi)
    mask = (R >= inner_radius) & (R <= outer_radius) & (
        (Theta_rotated <= critical_angle / 2) |
        ((Theta_rotated >= 0.5 * np.pi - critical_angle / 2) & (Theta_rotated <= 0.5 * np.pi + critical_angle / 2)) |
        ((Theta_rotated >= np.pi - critical_angle / 2) & (Theta_rotated <= np.pi + critical_angle / 2)) |
        ((Theta_rotated >= 1.5 * np.pi - critical_angle / 2) & (Theta_rotated <= 1.5 * np.pi + critical_angle / 2)) |
        (Theta_rotated >= 2 * np.pi - critical_angle / 2)
    )
    circular_source = np.zeros((aperture_size, aperture_size))
    circular_source[mask] = np.exp(-((R[mask] - inner_radius)**2) / (2 * sigma**2))
    circular_source /= circular_source.max()
    start_x = (grid_size_x - aperture_size) // 2
    start_y = (grid_size_y - aperture_size) // 2
    source[start_x:start_x + aperture_size, start_y:start_y + aperture_size] = circular_source
    return source

# Optical simulation
def optical_simulation(mask, source):
    mask_ft = fftshift(fft2(mask))
    source_ft = fftshift(fft2(source))
    result_ft = mask_ft * source_ft
    result = np.abs(ifft2(result_ft)) ** 2
    return result / result.max()

# Simulation parameters
inner_radius = np.random.randint(0, 6, 1500) / 10
outer_radius = inner_radius + np.random.randint(1, 5) / 10
critical_angles = np.random.randint(0, 90, 1500) * np.pi / 180
sigma = 1

# Generate mask and light source
mask = create_mask(grid_size_x, grid_size_y)

# Simulate and store results
all_data = []
high_data = []
low_data = []
abs_val = []
high_grad = []
low_grad = []

from tqdm import tqdm
import numpy as np

# Initialize all_data
all_data = []

# Loop through the parameter combinations
for inner_radius, outer_radius, critical_angle in tqdm(zip(inner_radius, outer_radius, critical_angles), total=1500):
    if outer_radius <= inner_radius:
        continue  # Skip invalid configurations
    
    # Generate light source and simulate
    light_source = create_light_source(grid_size_x, grid_size_y, inner_radius, outer_radius, critical_angle, sigma)
    result = optical_simulation(mask, light_source)
    
    # Compute the median value
    middle_val = np.median(np.round(result, 0))
    
    # Split results into high and low values
    high_vals = result[result >= middle_val]
    low_vals = result[result < middle_val]
    
    # Calculate standard deviations
    high_std_dev = round(np.std(high_vals), 2)
    low_std_dev = round(np.std(low_vals), 2)
    
    # Compute gradients (flatten arrays if needed)
    high_grad = np.mean(np.gradient(high_vals.flatten())) if high_vals.size > 1 else 0
    low_grad = np.mean(np.gradient(low_vals.flatten())) if low_vals.size > 1 else 0

    
    # Compute absolute difference (max - min)
    abs_val = result.max() - result.min()
    
    # Append the results
    all_data.append({
        "Inner Radius": inner_radius,
        "Outer Radius": outer_radius,
        "Sigma": sigma,
        "Critical Angle (degrees)": np.degrees(critical_angle),
        "High Standard Deviation": high_std_dev,
        "High Gradient": high_grad,
        "Low Standard Deviation": low_std_dev,
        "Low Gradient": low_grad,
        "Abs": abs_val
    })

# Save to CSV
output_dir = "D:/myData/研究所/先進微影/期末"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "output_simulation_Cqua_data2.csv")
df = pd.DataFrame(all_data)
df.to_csv(csv_path, index=False)
print(f"Data exported to {csv_path}")
