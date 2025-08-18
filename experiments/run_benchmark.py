# experiments/run_benchmark.py
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

from nemot_solver.solver import MOT_MC_Solver

# ===================================================================
# 1. ALGORITHM IMPLEMENTATIONS FOR BENCHMARKING
# ===================================================================

# --- Sinkhorn Algorithm (Brute-force baseline) ---
def run_one_sinkhorn_iteration(N_electrons, M_grid_size, dim, device):
    """
    Simulates one iteration of Sinkhorn by creating the massive cost tensor.
    This is designed to measure the time/memory bottleneck.
    """
    shape = tuple([M_grid_size] * N_electrons)
    
    # This tensor is the memory bottleneck.
    # We use torch.empty to just allocate memory without initializing,
    # which is faster and sufficient for a memory benchmark.
    try:
        cost_tensor = torch.empty(shape, device=device)
    except RuntimeError as e:
        # This will catch CUDA out of memory errors during allocation
        print(f"  Sinkhorn OOM on allocation for N={N_electrons}, M={M_grid_size}")
        raise e # Re-raise to be caught by measure_performance

    # The rest of the operations are computationally less significant
    # but included for a more complete time measurement.
    p_scaling_factors = [torch.randn(M_grid_size, device=device) for _ in range(N_electrons)]
    scaled_tensor = cost_tensor
    for axis in range(N_electrons):
        view_shape = [1] * N_electrons
        view_shape[axis] = M_grid_size
        scaler = p_scaling_factors[axis].view(view_shape)
        scaled_tensor = scaled_tensor * scaler
        
    # Example of a marginal computation
    if N_electrons > 1:
        axes_to_sum = list(range(1, N_electrons))
        marginal = torch.sum(scaled_tensor, dim=axes_to_sum)


# ===================================================================
# 2. BENCHMARKING UTILITIES
# ===================================================================

def measure_performance(func, device):
    # Warm-up run
    try:
        func()
    except RuntimeError: # Catch OOM on warm-up
        pass
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    try:
        # The actual timed run
        func()
    except RuntimeError:
        # If it fails here, it's a genuine OOM
        return float('nan')
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    return time.time() - start_time

# ===================================================================
# 3. EXPERIMENT RUNNERS & PLOTTING
# ===================================================================

def run_experiment_vs_n(s_total, m_total, b2_val, n_range, device):
    results = []
    print(f"\n--- Running Benchmark vs. N (Total Samples={s_total}, B2={b2_val}) ---")
    nemot_dataset = torch.randn(s_total, 2)
    
    for n in n_range:
        print(f"Testing N = {n}...")
        nemot_params = {
            'name': f'NEMOT N={n}', 'N_electrons': n, 'particle_dim': 2, 'epochs': 1, 
            'batch_size_support': 256, 'batch_size_tuples': b2_val, 
            'lr': 1e-4, 'eps_regularization': 0.01
        }
        nemot_solver = MOT_MC_Solver(nemot_params, device)
        
        nn_time = measure_performance(lambda: nemot_solver.train_potential_network(nemot_dataset), device=device)
        nn_mem = (b2_val * n * n) * 4 / (1024**2) # Memory for pairwise cost matrix
        print(f"  NEMOT-MC (1 Epoch): Time={nn_time:.4f}s, Theoretical Peak Mem={nn_mem:.2f}MB")
        
        sh_time = measure_performance(lambda: run_one_sinkhorn_iteration(n, m_total, 2, device), device=device)
        sh_mem = (m_total ** n) * 4 / (1024**2) # Memory for the full cost tensor
        print(f"  Sinkhorn (1 Iter): Time={sh_time:.4f}s, Theoretical Peak Mem={sh_mem:.2f}MB")
        
        results.append({'N': n, 'NMOT Time (s)': nn_time, 'NMOT Memory (MB)': nn_mem, 'Sinkhorn Time (s)': sh_time, 'Sinkhorn Memory (MB)': sh_mem})
    return pd.DataFrame(results)

def run_experiment_vs_m_s(n_val, b2_val, s_m_range, device):
    results = []
    print(f"\n--- Running Benchmark vs. Dataset Size (N={n_val}, B2={b2_val}) ---")
    nemot_params = {
        'name': f'NEMOT N={n_val}', 'N_electrons': n_val, 'particle_dim': 2, 'epochs': 1, 
        'batch_size_support': 256, 'batch_size_tuples': b2_val, 'lr': 1e-4, 'eps_regularization': 0.01
    }
    nemot_solver = MOT_MC_Solver(nemot_params, device)
    
    for s_m in s_m_range:
        print(f"Testing Total Samples = {s_m}...")
        nemot_dataset = torch.randn(s_m, 2)

        nn_time = measure_performance(lambda: nemot_solver.train_potential_network(nemot_dataset), device=device)
        nn_mem = (b2_val * n_val * n_val) * 4 / (1024**2)
        print(f"  NEMOT-MC (1 Epoch): Time={nn_time:.4f}s, Theoretical Peak Mem={nn_mem:.2f}MB")
        
        sh_time = measure_performance(lambda: run_one_sinkhorn_iteration(n_val, s_m, 2, device), device=device)
        sh_mem = (s_m ** n_val) * 4 / (1024**2)
        print(f"  Sinkhorn (1 Iter): Time={sh_time:.4f}s, Theoretical Peak Mem={sh_mem:.2f}MB")
        
        results.append({'M_total': s_m, 'NMOT Time (s)': nn_time, 'NMOT Memory (MB)': nn_mem, 'Sinkhorn Time (s)': sh_time, 'Sinkhorn Memory (MB)': sh_mem})
    return pd.DataFrame(results)

def plot_results(df, x_col, title_prefix, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{title_prefix}: Performance Comparison", fontsize=16)
    
    axes[0].plot(df[x_col], df['NMOT Time (s)'], 'o-', label='NMOT (1 Full Epoch)', color='blue')
    axes[0].plot(df[x_col], df['Sinkhorn Time (s)'], 's--', label='Sinkhorn (1 Iteration)', color='red')
    axes[0].set_title('Runtime'); axes[0].set_xlabel(x_col); axes[0].set_ylabel('Time (seconds)'); axes[0].set_yscale('log'); axes[0].grid(True, which="both", ls="--"); axes[0].legend()
    
    axes[1].plot(df[x_col], df['NMOT Memory (MB)'], 'o-', label='NMOT (Peak per Step)', color='blue')
    axes[1].plot(df[x_col], df['Sinkhorn Memory (MB)'], 's--', label='Sinkhorn (Peak)', color='red')
    axes[1].set_title('Theoretical Peak Memory Usage'); axes[1].set_xlabel(x_col); axes[1].set_ylabel('Peak Memory (MB)'); axes[1].set_yscale('log'); axes[1].grid(True, which="both", ls="--"); axes[1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if not os.path.exists('plots'): os.makedirs('plots')
    plt.savefig(filename); print(f"Plot saved to {filename}"); plt.close()

def run(device):
    """Main execution block for the benchmark."""
    
    # --- Experiment 1: Vary N ---
    n_range_exp1 = [2, 3, 4, 5, 6, 8, 10, 12]
    # NOTE: Sinkhorn m_total is small to avoid immediate OOM. 
    # This gives it a "best case" scenario.
    results_vs_n = run_experiment_vs_n(s_total=2048, m_total=64, b2_val=1024, n_range=n_range_exp1, device=device)
    print("\nResults vs. N:"); print(results_vs_n.to_string())
    plot_results(results_vs_n, 'N', 'Scaling vs. Number of Electrons (N)', 'plots/benchmark_nmot_vs_N.png')

    # --- Experiment 2: Vary Total Dataset Size ---
    n_fixed_exp2 = 3
    s_m_range_exp2 = [64, 128, 256, 512, 1024]
    results_vs_m_s = run_experiment_vs_m_s(n_val=n_fixed_exp2, b2_val=1024, s_m_range=s_m_range_exp2, device=device)
    print("\nResults vs. Total Dataset Size:"); print(results_vs_m_s.to_string())
    plot_results(results_vs_m_s, 'M_total', f'Scaling vs. Dataset/Grid Size (N={n_fixed_exp2})', 'plots/benchmark_nmot_vs_M_total.png')