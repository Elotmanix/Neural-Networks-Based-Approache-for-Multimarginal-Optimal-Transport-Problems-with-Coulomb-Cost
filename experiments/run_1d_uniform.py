# experiments/run_1d_uniform.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from nemot_solver.solver import MOT_Symmetric_Potential_Alg_Efficient
from nemot_solver.data_utils import generate_uniform_1d_samples, f_exact_uniform_1d, derive_f_learned_nd

def run(device):
    """Runs the 1D Uniform Density experiment."""
    
    # --- Configuration ---
    config = {
        'title': "Experiment for 1D Uniform Density",
        'savename': "plots/experiment_1d_uniform.png",
        'params': {
            'name': "Uniform 1D", 'N_electrons': 2, 'particle_dim': 1, 'epochs': 800, 
            'batch_size_support': 256, 'eps_regularization': 0.01, 'lr': 1e-4, 
            'K_neurons_phi': 512, 'deeper_phi': True, 'output_scale_phi': 5.0
        },
        'a': 2.0,
        'plot_range': [-1.0, 1.0],
    }

    # --- Training ---
    dataset = generate_uniform_1d_samples(100000, a=config['a'])
    solver = MOT_Symmetric_Potential_Alg_Efficient(config['params'], device)
    solver.train_potential_network(dataset)

    # --- Plotting ---
    print(f"\nGenerating plots for {config['params']['name']}...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    ((ax_tl, ax_tr), (ax_bl, ax_br)) = axes
    fig.suptitle(config['title'], fontsize=16)
    fig.tight_layout(pad=4.0)

    x_range = config['plot_range']
    x_plot_tensor = torch.linspace(x_range[0], x_range[1], 500).unsqueeze(-1)
    x_plot_np = x_plot_tensor.cpu().numpy().squeeze()
    
    # --- Top-Left: Kantorovich Potential ---
    solver.potential_model.eval()
    with torch.no_grad():
        u_vals = solver.potential_model(x_plot_tensor.to(device)).cpu().numpy()
    ax_tl.plot(x_plot_np, u_vals, color='black')
    ax_tl.set_title('Learned Kantorovich Potential')
    ax_tl.set_xlabel('x')
    ax_tl.set_ylabel('u(x)')

    # --- Top-Right: Co-motion function ---
    f_learned = derive_f_learned_nd(solver.potential_model, x_plot_tensor, x_plot_tensor, device).numpy()
    f_exact = f_exact_uniform_1d(x_plot_np, a=config['a'])
    
    ax_tr.plot(x_plot_np, f_exact, 'r--', linewidth=3, label='Analytical')
    ax_tr.plot(x_plot_np, f_learned.squeeze(), 'b-', label='Learned')
    ax_tr.set_title('Co-motion Function')
    ax_tr.set_xlabel('x')
    ax_tr.set_ylabel('f(x)')
    ax_tr.legend()
    ax_tr.axis('equal')

    # --- Bottom Panels: Transport plan γ ---
    gamma_res = 100
    x1, x2 = np.meshgrid(np.linspace(x_range[0], x_range[1], gamma_res), np.linspace(x_range[0], x_range[1], gamma_res))
    u_on_grid = np.interp(x1.ravel(), x_plot_np, u_vals.squeeze())
    u2_on_grid = np.interp(x2.ravel(), x_plot_np, u_vals.squeeze())
    cost = 1.0 / (np.abs(x1.ravel() - x2.ravel()) + 1e-9)
    exponent = (u_on_grid + u2_on_grid - cost) / config['params']['eps_regularization']
    gamma = np.exp(exponent - np.max(exponent))
    gamma_grid = gamma.reshape(gamma_res, gamma_res)

    extent = [x_range[0], x_range[1], x_range[0], x_range[1]]
    ax_br.imshow(gamma_grid, origin='lower', extent=extent, cmap='gray_r', aspect='auto')
    ax_br.set_title('Support of γ')
    ax_br.set_xlabel('x₁')
    ax_br.set_ylabel('x₂')
    ax_br.axis('equal')
    
    axes[1, 0] = fig.add_subplot(2, 2, 3, projection='3d')
    ax_bl = axes[1, 0]
    ax_bl.plot_surface(x1, x2, gamma_grid, cmap='gray_r', edgecolor='none')
    ax_bl.set_title('Transport plan γ')
    ax_bl.set_xlabel('x₁')
    ax_bl.set_ylabel('x₂')
    ax_bl.view_init(elev=40, azim=-135)

    if not os.path.exists('plots'): os.makedirs('plots')
    plt.savefig(config['savename'])
    print(f"Plot saved to {config['savename']}")
    plt.show()
