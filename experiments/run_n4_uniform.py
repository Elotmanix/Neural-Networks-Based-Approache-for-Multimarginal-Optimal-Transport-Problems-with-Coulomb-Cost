# experiments/run_n4_uniform.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from nemot_solver.solver import MOT_MC_Solver
from nemot_solver.data_utils import u_exact_n4

def run(device):
    """Runs the 1D Uniform Density experiment for N=4."""
    
    params = {
        'name': 'N=4 Uniform 1D', 'N_electrons': 4, 'particle_dim': 1,
        'epochs': 1000, 'batch_size_support': 32, 'batch_size_tuples': 1024,
        'eps_regularization': 0.005, 'lr': 1e-4, 'K_neurons_phi': 512,
        'deeper_phi': True, 'schedule_lr': True, 'schedule_patience': 50,
        'schedule_gamma': 0.5, 'clip_grads': True, 'max_grad_norm': 1.0,
        'output_scale_phi': 15.0
    }

    dataset = torch.linspace(0.0, 1.0, 5000).unsqueeze(-1)
    solver = MOT_MC_Solver(params, device)
    solver.train_potential_network(dataset)

    # --- Plotting ---
    print("\nGenerating plots for N=4 case...")
    solver.potential_model.eval()
    x_plot = torch.linspace(0.0, 1.0, 500).unsqueeze(-1)
    with torch.no_grad():
        u_nemot = solver.potential_model(x_plot.to(device)).cpu().numpy()

    u_exact = u_exact_n4(x_plot.squeeze().numpy())

    # Shift potentials to align at the center for comparison
    center_idx = np.argmin(np.abs(x_plot.squeeze().numpy() - 0.5))
    u_nemot_shifted = u_nemot - u_nemot[center_idx]
    u_exact_shifted = u_exact - u_exact[center_idx]
    
    rel_l2_error = np.linalg.norm(u_nemot_shifted - u_exact_shifted) / np.linalg.norm(u_exact_shifted)
    print(f"N=4 Relative L2 Error: {rel_l2_error:.4f}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x_plot.numpy(), u_exact_shifted, 'r--', linewidth=4, label='Analytical (N=4)')
    ax.plot(x_plot.numpy(), u_nemot_shifted, 'b-', linewidth=2.5, label='Learned (N=4)')
    ax.set_title('Kantorovich Potential for N=4, d=1 (Shifted)', fontsize=16)
    ax.set_xlabel('Position x', fontsize=12)
    ax.set_ylabel('u(x)', fontsize=12)
    ax.legend(fontsize=12)
    
    if not os.path.exists('plots'): os.makedirs('plots')
    plt.savefig('plots/experiment_n4_uniform.png')
    print("Plot saved to plots/experiment_n4_uniform.png")
    plt.show()