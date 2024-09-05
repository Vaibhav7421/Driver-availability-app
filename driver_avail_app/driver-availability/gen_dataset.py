import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
data = {
    'driver_id': np.arange(1, n_samples + 1),
    'hours_logged': np.random.uniform(0, 10, n_samples),
    'distance_driven': np.random.uniform(0, 20, n_samples),
    'trips_completed': np.random.randint(0, 50, n_samples),
    'rating': np.random.uniform(2, 5, n_samples),
    'available': np.random.randint(0, 2, n_samples)  # 0 for not available, 1 for available
}

# Create DataFrame
df = pd.DataFrame(data)

# Save dataset
df.to_csv('driver_availability_dataset.csv', index=False)
