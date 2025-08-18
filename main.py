# main.py
import torch
import sys
import os

# Import the experiment running functions
from experiments import (
    run_1d_uniform, run_1d_triangular, 
    run_2d_radial, run_3d_radial,
    run_n3_uniform, run_n4_uniform,
    run_benchmark  # <-- Add new import
)

def main():
    """Main function to run experiments."""
    if len(sys.argv) != 2:
        print("Usage: python main.py <experiment_name>")
        print("Available experiments: 1d_uniform, 1d_triangular, 2d_radial, 3d_radial, n3_uniform, n4_uniform, benchmark")
        sys.exit(1)

    experiment_name = sys.argv[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Match experiment name to the function
    if experiment_name == "1d_uniform":
        run_1d_uniform.run(device)
    elif experiment_name == "1d_triangular":
        run_1d_triangular.run(device)
    elif experiment_name == "2d_radial":
        run_2d_radial.run(device)
    elif experiment_name == "3d_radial":
        run_3d_radial.run(device)
    elif experiment_name == "n3_uniform":
        run_n3_uniform.run(device)
    elif experiment_name == "n4_uniform":
        run_n4_uniform.run(device)
    # --- Add new case ---
    elif experiment_name == "benchmark":
        run_benchmark.run(device)
    # ---------------------
    else:
        print(f"Error: Unknown experiment '{experiment_name}'")
        print("Available experiments: 1d_uniform, 1d_triangular, 2d_radial, 3d_radial, n3_uniform, n4_uniform, benchmark")
        sys.exit(1)

if __name__ == "__main__":
    main()