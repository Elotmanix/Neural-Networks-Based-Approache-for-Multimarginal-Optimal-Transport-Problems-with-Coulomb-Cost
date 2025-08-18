# nemot_solver/solver.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ... (Net_EOT and MOT_Symmetric_Potential_Alg_Efficient classes are unchanged) ...
class Net_EOT(nn.Module):
    # ...
    def __init__(self, dim, K, deeper=True, output_scale=10.0):
        super(Net_EOT, self).__init__()
        self.output_scale = output_scale
        layers = [nn.Linear(dim, K), nn.ReLU()]
        if deeper:
            layers.extend([nn.Linear(K, K), nn.ReLU(), nn.Linear(K, K), nn.ReLU()])
        layers.append(nn.Linear(K, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.tanh(self.net(x)).squeeze(-1) * self.output_scale
class MOT_Symmetric_Potential_Alg_Efficient(nn.Module):
    # ...
    def __init__(self, params: dict, device: torch.device):
        super(MOT_Symmetric_Potential_Alg_Efficient, self).__init__()
        self.params = params
        self.N_electrons = params['N_electrons']
        self.particle_dim = params['particle_dim']
        self.eps_regularization = params['eps_regularization']
        self.num_epochs = params['epochs']
        self.batch_size_support = params['batch_size_support']
        self.device = device
        self.potential_model = Net_EOT(
            dim=self.particle_dim,
            K=params.get('K_neurons_phi', 512),
            deeper=params.get('deeper_phi', True),
            output_scale=params.get('output_scale_phi', 5.0)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.potential_model.parameters(), lr=params['lr'])
        self.best_dual_obj = -float('inf')
        self.best_model_state = None
    def _calculate_batch_cost(self, x: torch.Tensor) -> torch.Tensor:
        p1 = x.unsqueeze(1)
        p2 = x.unsqueeze(0)
        dist_sq = torch.sum((p1 - p2)**2, dim=-1)
        return 1.0 / (torch.sqrt(dist_sq) + 1e-9)
    def _calculate_emot_exp_term(self, sum_phi_tensor: torch.Tensor, cost_tensor: torch.Tensor):
        alpha = (sum_phi_tensor - cost_tensor) / self.eps_regularization
        alpha_max = torch.max(alpha)
        if torch.isinf(alpha_max): return torch.tensor(float('inf'), device=alpha.device)
        exp_term = torch.mean(torch.exp(alpha - alpha_max))
        return torch.log(exp_term) + alpha_max
    def train_potential_network(self, dataset_rho_support: torch.Tensor):
        torch_dataset = TensorDataset(dataset_rho_support)
        data_loader = DataLoader(torch_dataset, batch_size=self.batch_size_support, shuffle=True, drop_last=True)
        self.potential_model.train()
        print(f"\n--- Starting Training for {self.params['name']} ---")
        for epoch in range(self.num_epochs):
            epoch_dual_obj = []
            for (batch_tuple,) in data_loader:
                current_batch_rho_support = batch_tuple.to(self.device)
                self.optimizer.zero_grad()
                phi_on_batch = self.potential_model(current_batch_rho_support)
                sum_mean_phi_term = self.N_electrons * torch.mean(phi_on_batch)
                sum_phi_tensor_batch = phi_on_batch.view(-1, 1) + phi_on_batch.view(1, -1)
                cost_tensor_batch = self._calculate_batch_cost(current_batch_rho_support)
                log_e_term_val = self._calculate_emot_exp_term(sum_phi_tensor_batch, cost_tensor_batch)
                if torch.isinf(log_e_term_val) or torch.isnan(log_e_term_val): continue
                dual_objective = sum_mean_phi_term - self.eps_regularization * log_e_term_val
                loss = -dual_objective
                if torch.isinf(loss) or torch.isnan(loss): continue
                loss.backward()
                self.optimizer.step()
                epoch_dual_obj.append(dual_objective.item())
            avg_dual_obj = np.mean(epoch_dual_obj) if epoch_dual_obj else -np.inf
            if (epoch + 1) % 100 == 0: print(f"  Epoch {epoch+1}/{self.num_epochs} | Avg Dual Obj: {avg_dual_obj:.6f}")
            if avg_dual_obj > self.best_dual_obj:
                self.best_dual_obj = avg_dual_obj
                self.best_model_state = self.potential_model.state_dict()
        if self.best_model_state:
            self.potential_model.load_state_dict(self.best_model_state)
            print(f"Training finished. Best dual objective: {self.best_dual_obj:.6f}")


# ===================================================================
# ==== SCALABLE SOLVER FOR N > 2 ====
# ===================================================================
class MOT_MC_Solver(nn.Module):
    """Scalable Monte Carlo style solver for indistinguishable particles."""
    def __init__(self, params: dict, device: torch.device):
        super(MOT_MC_Solver, self).__init__()
        self.params = params
        self.N_electrons = params['N_electrons']
        self.particle_dim = params['particle_dim']
        self.eps_regularization = params['eps_regularization']
        self.num_epochs = params['epochs']
        self.batch_size_support = params['batch_size_support']
        self.batch_size_tuples = params.get('batch_size_tuples', 1024)
        self.device = device
        self.potential_model = Net_EOT(
            dim=self.particle_dim, K=params.get('K_neurons_phi', 512),
            deeper=params.get('deeper_phi', True),
            output_scale=params.get('output_scale_phi', 10.0)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.potential_model.parameters(), lr=params['lr'])

        if params.get('schedule_lr', False):
            # ===================================================================
            # ==== FIX IS HERE: REMOVED `verbose=False` ====
            # ===================================================================
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=params.get('schedule_gamma', 0.5),
                patience=params.get('schedule_patience', 20)
            )
        else:
            self.scheduler = None
        self.best_dual_obj = -float('inf')
        self.best_model_state = None

    # ... (the rest of the MOT_MC_Solver class is unchanged) ...
    def pairwise_coulomb(self, positions):
        diff = positions[:, :, None, :] - positions[:, None, :, :]
        dist = torch.norm(diff, dim=-1)
        mask = torch.eye(self.N_electrons, device=self.device).bool()[None, :, :]
        dist = dist.masked_fill(mask, float('inf'))
        return 1.0 / (dist + 1e-9)
    def train_potential_network(self, dataset_rho_support: torch.Tensor):
        torch_dataset = TensorDataset(dataset_rho_support)
        data_loader = DataLoader(torch_dataset, batch_size=self.batch_size_support, shuffle=True, drop_last=True)
        self.potential_model.train()
        print(f"\n--- Scalable MC Training for N={self.N_electrons} | B2={self.batch_size_tuples} ---")
        for epoch in range(self.num_epochs):
            epoch_losses = []
            for (batch_tuple,) in data_loader:
                support_points = batch_tuple.to(self.device)
                S = support_points.shape[0]
                if S < self.N_electrons: continue
                self.optimizer.zero_grad()
                phi_single = self.potential_model(support_points)
                sum_mean_phi_term = self.N_electrons * torch.mean(phi_single)
                B2 = self.batch_size_tuples
                idx = torch.randint(0, S, (B2, self.N_electrons), device=self.device)
                tuples = support_points[idx]
                phi_tuples = self.potential_model(tuples.view(-1, self.particle_dim)).view(B2, self.N_electrons)
                sum_phi = torch.sum(phi_tuples, dim=1)
                pairwise = self.pairwise_coulomb(tuples)
                cost_vals = pairwise.sum(dim=(1, 2)) * 0.5
                alpha = (sum_phi - cost_vals) / self.eps_regularization
                alpha_max = torch.max(alpha)
                exp_term = torch.mean(torch.exp(alpha - alpha_max))
                log_e_term_val = torch.log(exp_term) + alpha_max
                dual_objective = sum_mean_phi_term - self.eps_regularization * log_e_term_val
                loss = -dual_objective
                if torch.isfinite(loss):
                    loss.backward()
                    if self.params.get('clip_grads', False):
                        torch.nn.utils.clip_grad_norm_(
                            self.potential_model.parameters(), self.params.get('max_grad_norm', 1.0)
                        )
                    self.optimizer.step()
                    epoch_losses.append(loss.item())
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                if (epoch + 1) % 50 == 0: print(f"  Epoch {epoch+1}/{self.num_epochs} | Avg. Loss: {avg_loss:.6f}")
                if (-avg_loss) > self.best_dual_obj:
                    self.best_dual_obj = -avg_loss
                    self.best_model_state = self.potential_model.state_dict()
                if self.scheduler: self.scheduler.step(avg_loss)
        if self.best_model_state:
            self.potential_model.load_state_dict(self.best_model_state)
            print(f"Training finished. Best dual objective: {self.best_dual_obj:.6f}")
        else:
            print("Training finished, but no valid model state was saved.")