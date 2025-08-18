# nemot_solver/data_utils.py

# ... (all previous functions in this file remain unchanged) ...
import torch
import numpy as np

# --- Data Generation ---
# ...
def generate_uniform_1d_samples(n_samples, a=2.0):
    return (torch.rand(n_samples) * a - a / 2).unsqueeze(-1)
def generate_triangular_1d_samples(n_samples, a=1.0):
    u = np.random.rand(n_samples)
    samples = np.where(u < 0.5, a * np.sqrt(2 * u) - a, a - a * np.sqrt(2 * (1 - u)))
    return torch.tensor(samples, dtype=torch.float32).unsqueeze(-1)
def generate_uniform_disk_samples(n_samples, radius=1.0):
    r = radius * torch.sqrt(torch.rand(n_samples))
    theta = 2 * np.pi * torch.rand(n_samples)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)
def generate_uniform_ball_samples(n_samples, radius=1.0):
    r = radius * torch.pow(torch.rand(n_samples), 1/3)
    theta = 2 * np.pi * torch.rand(n_samples)
    phi = torch.acos(1 - 2 * torch.rand(n_samples))
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


# --- Analytical Co-motion Functions ---
# ...
def f_exact_uniform_1d(x, a=2.0):
    half_a = a / 2
    return np.piecewise(x, [x >= 0, x < 0], [lambda x: x - half_a, lambda x: x + half_a])


def f_exact_triangular_1d(x, a=1.0):
    x_abs = np.abs(x)
    sign = np.sign(x)
    sign[sign == 0] = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        val = sign * (np.sqrt(2*a*x_abs - x_abs**2) - a)
    return val
def a_r_exact_disk_2d(r):
    return np.sqrt(1 - r**2)
def a_r_exact_ball_3d(r):
    return np.power(1 - r**3, 1/3)

# ===================================================================
# ==== NEW ANALYTICAL POTENTIALS FOR N > 2 ====
# ===================================================================
def u_exact_n3(x):
    """Analytical potential for N=3, d=1, uniform density on [0,1]. From paper Eq (27)."""
    return np.piecewise(x,
                        [x < 1/3, (x >= 1/3) & (x < 2/3), x >= 2/3],
                        [lambda x: (45/4) * x,
                         lambda x: 15/4,
                         lambda x: -(45/4) * x + 45/4])

def u_exact_n4(x):
    """Analytical potential for N=4, d=1, uniform density on [0,1]."""
    conds = [(x>=0)&(x<=1/4), (x>1/4)&(x<=1/2), (x>1/2)&(x<=3/4), (x>3/4)&(x<=1)]
    funcs = [
        lambda x: 21.77 * x + 2.5, lambda x: 4 * x + 6.9425,
        lambda x: -4 * x + 10.9425, lambda x: -21.77 * x + 24.27
    ]
    return np.piecewise(x, conds, funcs)
# ===================================================================

# --- Co-motion Derivation & Grid Generation ---
# ...
def derive_f_learned_nd(model, source_points, all_points, device):
    source_points = source_points.to(device)
    all_points = all_points.to(device)
    model.eval()
    dest_points = torch.zeros_like(source_points) 
    with torch.no_grad():
        u_on_all_points = model(all_points)
        for i, src in enumerate(source_points):
            cost_vec = 1.0 / (torch.norm(src - all_points, p=2, dim=1) + 1e-9)
            objective_vec = u_on_all_points - cost_vec
            best_j_idx = torch.argmax(objective_vec)
            dest_points[i] = all_points[best_j_idx]
    return dest_points.cpu()
def create_regular_ball_grid(resolution=50):
    coords = np.linspace(-1.0, 1.0, resolution)
    xv, yv, zv = np.meshgrid(coords, coords, coords)
    grid = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
    radii = np.linalg.norm(grid, axis=1)
    ball_points = grid[radii <= 1.0]
    return torch.tensor(ball_points, dtype=torch.float32)