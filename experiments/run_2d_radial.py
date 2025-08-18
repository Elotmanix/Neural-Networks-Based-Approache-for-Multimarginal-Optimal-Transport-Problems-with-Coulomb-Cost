# experiments/run_2d_radial.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from nemot_solver.solver import MOT_Symmetric_Potential_Alg_Efficient
from nemot_solver.data_utils import generate_uniform_disk_samples, a_r_exact_disk_2d, derive_f_learned_nd

def run(device):
    """Runs the 2D Radially Symmetric experiment."""
    
    params = {
        'name': 'Reproduction of Figure 5 (2D Radial)',
        'N_electrons': 2, 'particle_dim': 2, 'epochs': 800,
        'batch_size_support': 256, 'eps_regularization': 0.002, 'lr': 5e-5,
        'K_neurons_phi': 512, 'deeper_phi': True, 'output_scale_phi': 5.0
    }

    dataset = generate_uniform_disk_samples(100000, radius=1.0)
    solver = MOT_Symmetric_Potential_Alg_Efficient(params, device)
    solver.train_potential_network(dataset)

    print("\nGenerating plots for 2D radial case...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ((ax_tl, ax_tr), (ax_bl, ax_br)) = axes
    fig.suptitle('Results for N=2, d=2 Uniform Density (Unit Disk)', fontsize=16)
    fig.tight_layout(pad=4.0)

    # --- Top-Left: Kantorovich Potential ---
    solver.potential_model.eval()
    r_plot = torch.linspace(0, 1, 100).to(device)
    points_on_axis = torch.stack([r_plot, torch.zeros_like(r_plot)], dim=1)
    with torch.no_grad():
        v_r = solver.potential_model(points_on_axis).cpu()
    ax_tl.plot(r_plot.cpu(), -v_r, color='black')
    ax_tl.set_title('Kantorovich Potential')
    ax_tl.set_xlabel('r')
    ax_tl.set_ylabel('-v(r)')

    # --- Top-Right: Co-motion function ---
    source_points = dataset[torch.randperm(len(dataset))[:500]]
    dest_points = derive_f_learned_nd(solver.potential_model, source_points, dataset.to(device), device)
    source_radii = torch.norm(source_points, p=2, dim=1)
    dest_radii = torch.norm(dest_points, p=2, dim=1)
    sort_indices = torch.argsort(source_radii)
    
    analytical_r = np.linspace(0, 1, 100)
    analytical_a_r = a_r_exact_disk_2d(analytical_r)
    
    ax_tr.plot(analytical_r, analytical_a_r, 'r--', linewidth=3, label='Analytical')
    ax_tr.plot(source_radii[sort_indices].numpy(), dest_radii[sort_indices].numpy(), 'b-', linewidth=2, label='Learned')
    ax_tr.set_title('Radial Co-motion Function a(r)')
    ax_tr.set_xlabel('Source Radius r'); ax_tr.set_ylabel('Destination Radius a(r)')
    ax_tr.set_xlim(0, 1); ax_tr.set_ylim(0, 1)
    ax_tr.legend(); ax_tr.set_aspect('equal', adjustable='box')

    # --- Bottom Panels: Transport plan γ ---
    gamma_res = 100
    r1, r2 = np.meshgrid(np.linspace(1e-3, 1, gamma_res), np.linspace(1e-3, 1, gamma_res))
    v_r1 = np.interp(r1.ravel(), r_plot.cpu(), v_r)
    v_r2 = np.interp(r2.ravel(), r_plot.cpu(), v_r)
    cost_r = 1.0 / (r1.ravel() + r2.ravel())
    exponent = (v_r1 + v_r2 - cost_r) / params['eps_regularization']
    gamma = np.exp(exponent - np.max(exponent))
    gamma_grid = gamma.reshape(gamma_res, gamma_res)

    ax_br.imshow(gamma_grid, origin='lower', extent=[0, 1, 0, 1], cmap='gray_r', aspect='auto')
    ax_br.set_title('Support of γ'); ax_br.set_xlabel('r₁'); ax_br.set_ylabel('r₂')
    ax_br.set_aspect('equal', adjustable='box')
    
    axes[1,0] = fig.add_subplot(2, 2, 3, projection='3d')
    ax_bl = axes[1,0]
    ax_bl.plot_surface(r1, r2, gamma_grid, cmap='gray_r', edgecolor='none')
    ax_bl.set_title('Transport plan γ'); ax_bl.set_xlabel('r₁'); ax_bl.set_ylabel('r₂')
    ax_bl.view_init(elev=40, azim=-45)

    if not os.path.exists('plots'): os.makedirs('plots')
    plt.savefig('plots/experiment_2d_radial.png')
    print("Plot saved to plots/experiment_2d_radial.png")
    plt.show()
