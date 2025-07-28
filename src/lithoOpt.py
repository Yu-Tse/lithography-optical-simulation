from scipy.optimize import basinhopping
import pandas as pd
import numpy as np

# Define the objective function
def objective_function(params, data):
    inner_radius, outer_radius, critical_angle = params

    # Filter rows that match the given parameters (with tolerance)
    filtered_data = data[
        (np.isclose(data["Inner Radius"], inner_radius, atol=0.1)) &
        (np.isclose(data["Outer Radius"], outer_radius, atol=0.1)) &
        (np.isclose(data["Critical Angle (degrees)"], critical_angle, atol=0.1))
    ]

    # If no matching data found, return a large penalty value
    if filtered_data.empty:
        return 1e6

    # Calculate mean of high and low standard deviations and gradients
    high_std_dev = filtered_data["High Standard Deviation"].mean()
    high_grad = filtered_data["High Gradient"].mean()
    low_std_dev = filtered_data["Low Standard Deviation"].mean()
    low_grad = filtered_data["Low Gradient"].mean()

    # Objective: minimize total variation and gradient energy
    return high_std_dev + low_std_dev + high_grad**2 + low_grad**2

# Load data
data_path = "D:/myData/lithography/test.csv"  # 🔧 Change to your CSV path
data = pd.read_csv(data_path)

# Initial parameter guess (mean of dataset values)
initial_guess = [
    data["Inner Radius"].mean(),
    data["Outer Radius"].mean(),
    data["Critical Angle (degrees)"].mean()
]

# Parameter bounds
bounds = [
    (0.1, 0.9),     # Inner Radius
    (0.1, 1.0),     # Outer Radius
    (0, 90)         # Critical Angle in degrees
]

# Run basinhopping optimization
result = basinhopping(
    func=objective_function,
    x0=initial_guess,
    minimizer_kwargs={"args": (data,), "method": "L-BFGS-B", "bounds": bounds},
    niter=200
)

# Print results
print("Optimal Parameters:", result.x)
print("Objective Score (Total Std Dev + Gradient Energy):", result.fun)
