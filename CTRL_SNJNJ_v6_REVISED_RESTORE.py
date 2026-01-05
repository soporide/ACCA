
"""
CTRL_SNJNG_v6_full.py

Full runnable implementation of ACCA (SNJNG + DeepPA + DeepPER + CTRL with NODE + ITDE).

This file is designed to be:
- Executable end-to-end (no pseudocode / no TODO placeholders)
- Faithful to the key equations extracted from ACPS08022025_JOC.docx (ACCA paper draft)
- Practical: supports *linear-time* (O(N+|E|)) simulation/training loops by using sparse message passing
- Safe: includes numerical stability guards and test assertions

Key equations:
(1)  S_t = SNJNG(t)
(2)  A_t ∈ R^n
(3)  dS/dt = f_θ(S_t, A_t)
      with jumps: S(τ^+) = S(τ^-) + J_θ(S(τ^-), ξ)
(3)  Reward shaping: R_t = -γ1 L_t + γ2 PA_t + γ3 PER_t
(4)  V(S_t)=E[∫_t^∞ e^{-γ(τ-t)} R_τ dτ | S_t]
(5)  ITDE (piecewise): integral TD error with jump at τ ∈ (t, t+Δt)
(6)  Node jump: P_jump_i = 1 - exp(-λ_i Δt),  λ_i = f_θ(h_i) (softplus)
(7)  Degree update: k_i^+ = k_i^- + Δk_i, and adjacency update A^+ = Θ_jump(A^-, Δk_i)
(8)  Edge jump: P_jump_ij = 1 - exp(-λ_ij Δt), and A_ij^+ = 1 - A_ij^- when jump
(9)  DeepPER: P_ij = cos^2(h_i,h_j), T_ij = exp(-mean_d Δh^2 / (2 ξ_d^2)),
             P'_ij = P_ij + (1 - P_ij) T_ij,
             λ_ij = λ0 * T_ij * g_ϕ(h_i,h_j)
(10) DeepPA: attachment probability Π(k) ∝ k^α(t), with entanglement/superposition:
             α(t)=α0+Δα t + noise, and mixture weights γ_s(t) over α states.

Usage:
    python CTRL_SNJNJ_v3_full.py --run_tests
    python CTRL_SNJNJ_v3_full.py --demo

Notes:
- This is a research-grade *reference implementation* meant to be readable and modifiable.
- For large N, exact DeepPER vectorization is expensive; we provide a linear-time PER proxy
  (percolation threshold ζ_c and giant component fraction) and keep the exact vectorized DeepPER
  as an optional diagnostic for small graphs (N<=200).
"""

from __future__ import annotations

import argparse
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Optional visualization (matplotlib)
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


###############################################################################
#                                0. Helpers                                  #
###############################################################################

def set_seed(seed: int) -> None:
    """
    Global determinism + performance sanity for CPU workloads.
    For small/medium graphs on CPU, PyTorch's default high thread count can *hurt* performance
    dramatically (thread launch overhead). We cap threads to 1 here for predictable runtime.
    """
    # Performance guard for small ops (index_add/scatter_add in message passing)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        # If threads are already configured by the runtime, ignore.
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def softplus_stable(x: torch.Tensor) -> torch.Tensor:
    # stable softplus to ensure positivity
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.maximum(x, torch.zeros_like(x))


def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def expm1_stable(x: float) -> float:
    # stable exp(x)-1 for small x
    return math.expm1(x)


def analytic_discounted_integral(reward: float, dt: float, discount_rate: float) -> float:
    """
    Compute ∫_0^dt exp(-discount_rate * u) * reward du.
    If discount_rate == 0, returns reward*dt.
    """
    if dt <= 0:
        return 0.0
    if abs(discount_rate) < 1e-12:
        return float(reward) * float(dt)
    return float(reward) * (1.0 - math.exp(-discount_rate * dt)) / discount_rate


def topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, x.size)
    # argpartition for O(N)
    idx = np.argpartition(-x, k - 1)[:k]
    # sort descending for determinism
    idx = idx[np.argsort(-x[idx])]
    return idx


###############################################################################
#                     1. Dynamic graph representation                         #
###############################################################################

class AttackGraph:
    """
    Undirected dynamic graph with:
    - neighbor sets (for O(1) add/remove edge)
    - edge weights dict for optional weighted edges (e.g., jamming attenuation)
    """
    def __init__(self, N: int):
        self.N = N
        self.neigh: List[set[int]] = [set() for _ in range(N)]
        # store only (min(i,j),max(i,j)) keys
        self.w: Dict[Tuple[int, int], float] = {}

    def has_edge(self, i: int, j: int) -> bool:
        if i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        return (a, b) in self.w and self.w[(a, b)] > 0.0

    def edge_weight(self, i: int, j: int) -> float:
        a, b = (i, j) if i < j else (j, i)
        return float(self.w.get((a, b), 0.0))

    def add_edge(self, i: int, j: int, weight: float = 1.0) -> None:
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        self.w[(a, b)] = float(weight)
        self.neigh[i].add(j)
        self.neigh[j].add(i)

    def remove_edge(self, i: int, j: int) -> None:
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in self.w:
            del self.w[(a, b)]
        self.neigh[i].discard(j)
        self.neigh[j].discard(i)

    def flip_edge(self, i: int, j: int, weight_if_add: float = 1.0) -> None:
        if self.has_edge(i, j):
            self.remove_edge(i, j)
        else:
            self.add_edge(i, j, weight_if_add)

    def degrees(self) -> np.ndarray:
        return np.array([len(s) for s in self.neigh], dtype=np.float32)

    def num_edges(self) -> int:
        return len(self.w)

    def undirected_edges(self) -> List[Tuple[int, int, float]]:
        out = []
        for (i, j), w in self.w.items():
            out.append((i, j, float(w)))
        return out

    def to_dense_adjacency(self) -> np.ndarray:
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for (i, j), w in self.w.items():
            A[i, j] = w
            A[j, i] = w
        np.fill_diagonal(A, 0.0)
        return A

    def assert_symmetric(self) -> None:
        # structural symmetry is guaranteed by storage, but we validate neighbors
        for i in range(self.N):
            for j in self.neigh[i]:
                assert i in self.neigh[j], f"Asymmetry: {i}->{j} but not {j}->{i}"
                assert self.has_edge(i, j), f"Neighbor set has {i}-{j} but weight missing"

    def clone(self) -> "AttackGraph":
        """Deep copy for counterfactual tests / environment cloning."""
        g = AttackGraph(self.N)
        g.neigh = [set(s) for s in self.neigh]
        g.w = dict(self.w)
        return g


###############################################################################
#                            2. Sparse GCN                                    #
###############################################################################

def build_gcn_edges(graph: AttackGraph, jam_strength: float = 0.0, jam_scale: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build directed edge_index (2, E_dir) and normalized edge weights for a GCN layer:
        H^{l+1} = σ( D^{-1/2} Â D^{-1/2} H^l W^l )
    with Â = A + I (self loops included).
    Uses edge weights optionally attenuated by jamming.

    Complexity: O(N + |E|)
    """
    N = graph.N
    deg = graph.degrees()  # degrees for adjacency only
    # Use weighted degrees for normalization if edges have weights.
    # For stability, keep it simple: deg_w = sum(w_ij) + 1 for self-loop.
    deg_w = np.ones(N, dtype=np.float32)  # for self-loop
    for (i, j), w in graph.w.items():
        w_eff = float(w) * (1.0 - jam_scale * float(jam_strength))
        w_eff = max(w_eff, 0.0)
        deg_w[i] += w_eff
        deg_w[j] += w_eff

    d_inv_sqrt = 1.0 / np.sqrt(deg_w + 1e-12)
    d_inv_sqrt_t = torch.tensor(d_inv_sqrt, dtype=torch.float32)

    src_list = []
    dst_list = []
    w_list = []

    # self loops
    for i in range(N):
        src_list.append(i)
        dst_list.append(i)
        w_list.append(float(d_inv_sqrt[i] * 1.0 * d_inv_sqrt[i]))

    # undirected edges -> two directed edges
    for (i, j), w in graph.w.items():
        w_eff = float(w) * (1.0 - jam_scale * float(jam_strength))
        w_eff = max(w_eff, 0.0)
        if w_eff <= 0.0:
            continue
        norm_ij = float(d_inv_sqrt[i] * w_eff * d_inv_sqrt[j])
        norm_ji = float(d_inv_sqrt[j] * w_eff * d_inv_sqrt[i])
        src_list.append(j); dst_list.append(i); w_list.append(norm_ij)  # j -> i
        src_list.append(i); dst_list.append(j); w_list.append(norm_ji)  # i -> j

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.int64)
    edge_weight = torch.tensor(w_list, dtype=torch.float32)
    return edge_index, edge_weight


def spmm(edge_index: torch.Tensor, edge_weight: torch.Tensor, X: torch.Tensor, N: int) -> torch.Tensor:
    """
    Sparse message passing: out[dst] += edge_weight * X[src]
    edge_index: (2, E)
    edge_weight: (E,)
    X: (N, F)
    """
    src = edge_index[0]
    dst = edge_index[1]
    msg = X[src] * edge_weight.unsqueeze(-1)
    out = torch.zeros((N, X.size(1)), dtype=X.dtype, device=X.device)
    out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, X.size(1)), msg)
    return out


class SparseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        N = X.size(0)
        agg = spmm(edge_index, edge_weight, X, N)
        return self.lin(agg)


class SparseGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        assert num_layers >= 1
        layers = []
        if num_layers == 1:
            layers.append(SparseGCNLayer(in_dim, out_dim))
        else:
            layers.append(SparseGCNLayer(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(SparseGCNLayer(hidden_dim, hidden_dim))
            layers.append(SparseGCNLayer(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = X
        for li, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight)
            if li != len(self.layers) - 1:
                h = self.act(h)
        return h


###############################################################################
#                         3. SNJNG: neural intensities                         #
###############################################################################

class NodeIntensityNN(nn.Module):
    def __init__(self, h_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N, h_dim) -> (N,)
        out = self.net(h).squeeze(-1)
        return softplus_stable(out) + 1e-8


class EdgeIntensityNN(nn.Module):
    def __init__(self, h_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * h_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        # Undirected-graph symmetry: use order-invariant features so λ_ij == λ_ji.
        x = torch.cat([h_i + h_j, torch.abs(h_i - h_j)], dim=-1)
        out = self.net(x).squeeze(-1)
        return softplus_stable(out) + 1e-8


class DegreeDeltaCategorical(nn.Module):
    """
    Predict Δk as a categorical distribution over {-K, ..., K}.
    """
    def __init__(self, h_dim: int, K: int = 5, hidden: int = 64):
        super().__init__()
        self.K = int(K)
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * self.K + 1),
        )

    def sample(self, h: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        h: (h_dim,) tensor
        returns (delta_k:int, log_prob: torch.Tensor scalar)
        """
        logits = self.net(h.unsqueeze(0)).squeeze(0)  # (2K+1,)
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)
        delta_k = int(idx.item()) - self.K
        return delta_k, log_prob

    def log_prob(self, h: torch.Tensor, delta_k: int) -> torch.Tensor:
        """Log-probability of an observed Δk under the current categorical model."""
        dk = int(delta_k)
        if dk < -self.K or dk > self.K:
            # Outside support: return large negative log-prob (finite)
            return torch.tensor(-1e8, dtype=h.dtype, device=h.device)
        logits = self.net(h.unsqueeze(0)).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        idx = torch.tensor(dk + self.K, dtype=torch.int64, device=h.device)
        return dist.log_prob(idx)

    def mode(self, h: torch.Tensor) -> int:
        logits = self.net(h.unsqueeze(0)).squeeze(0)
        idx = int(torch.argmax(logits).item())
        return idx - self.K


###############################################################################
#                     4. DeepPA: entanglement/superposition                    #
###############################################################################

class EntanglementNN(nn.Module):
    """
    Output mixture weights γ_s(t) over S PA "states" (superposition).
    """
    def __init__(self, state_dim: int, S: int = 3, hidden: int = 64):
        super().__init__()
        self.S = int(S)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden, self.S),
        )

    def forward(self, s: torch.Tensor, t: float) -> torch.Tensor:
        # returns weights (S,)
        t_t = torch.tensor([t], dtype=s.dtype, device=s.device)
        x = torch.cat([s, t_t], dim=0)
        logits = self.net(x)
        return torch.softmax(logits, dim=0)


def pa_target_distribution(deg: np.ndarray, alpha_states: np.ndarray, gamma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Preferential attachment target distribution:
        p_j ∝ Σ_s γ_s * (deg_j + eps)^{α_s}
    deg: (N,)
    alpha_states: (S,)
    gamma: (S,)
    returns p: (N,) sum to 1
    """
    N = deg.size
    w = np.zeros(N, dtype=np.float64)
    for a, g in zip(alpha_states, gamma):
        w += float(g) * np.power(deg + eps, float(a))
    w_sum = w.sum()
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        # fallback uniform
        return np.ones(N, dtype=np.float64) / N
    return w / w_sum


def pa_target_logits_torch(
    deg: torch.Tensor,
    alpha_states: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Torch-stable logits for preferential attachment mixture (Eq. 6,10).

        Π_j(k,t) ∝ Σ_s γ_s(t) * (deg_j + eps)^{α_s(t)}

    We compute logits via log-sum-exp for numerical stability:
        log w_j = log Σ_s exp( log γ_s + α_s log(deg_j+eps) ).

    Args:
        deg: (N,) nonnegative degrees (no grad).
        alpha_states: (S,) PA exponents for superposition states (no grad).
        gamma: (S,) simplex weights from EntanglementNN (learned).
    Returns:
        log_w: (N,) logits (unnormalized log-weights).
    """
    # Ensure shapes
    deg = deg.to(dtype=torch.float32)
    alpha_states = alpha_states.to(dtype=torch.float32)
    gamma = gamma.to(dtype=torch.float32)

    log_deg = torch.log(deg.clamp_min(0.0) + eps)  # (N,)
    log_gamma = safe_log(gamma, eps=1e-12)          # (S,)
    # Broadcast to (S,N)
    terms = log_gamma.unsqueeze(1) + alpha_states.unsqueeze(1) * log_deg.unsqueeze(0)
    log_w = torch.logsumexp(terms, dim=0)
    # Defensive finite guard
    log_w = torch.where(torch.isfinite(log_w), log_w, torch.zeros_like(log_w))
    return log_w


def pa_target_distribution_torch(
    deg: torch.Tensor,
    alpha_states: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return Π(k) as a probability vector over nodes (N,)."""
    logits = pa_target_logits_torch(deg, alpha_states, gamma, eps=eps)
    # Softmax to get a valid distribution
    return torch.softmax(logits, dim=0)


###############################################################################
#                          5. DeepPER: tunneling                               #
###############################################################################

def cosine_squared(h_i: torch.Tensor, h_j: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = (h_i * h_j).sum()
    den = (h_i.norm() * h_j.norm()).clamp_min(eps)
    c = (num / den).clamp(-1.0, 1.0)
    return c * c


def tunneling_term(h_i: torch.Tensor, h_j: torch.Tensor, xi_d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    T_ij = exp( - mean_d [ (Δh_d)^2 / (2 ξ_d^2) ] )
    xi_d may be scalar tensor () or vector (D,)
    """
    dh = (h_i - h_j)
    if xi_d.numel() == 1:
        denom = 2.0 * (xi_d.item() ** 2 + eps)
        val = - (dh.pow(2).mean() / denom)
    else:
        # broadcast vector
        denom = 2.0 * (xi_d.pow(2) + eps)
        val = - ((dh.pow(2) / denom).mean())
    return torch.exp(val)


def adjusted_connection_probability(P_ij: torch.Tensor, T_ij: torch.Tensor) -> torch.Tensor:
    # P' = P + (1-P)T
    return P_ij + (1.0 - P_ij) * T_ij


###############################################################################
#                    6. CTRL: policy, value, NODE, jump map                     #
###############################################################################

class DynamicsNN(nn.Module):
    """
    f_θ(s,a,t) for NODE: ds/dt = f(s,a,t)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor, t: float) -> torch.Tensor:
        t_t = torch.tensor([t], dtype=s.dtype, device=s.device)
        x = torch.cat([s, a, t_t], dim=0)
        return self.net(x)


class JumpMapNN(nn.Module):
    """
    J_θ(s, ξ) returning Δs for instantaneous reset at jump time τ.
    ξ is a low-dimensional "mark" vector encoding event type and indices.
    """
    def __init__(self, state_dim: int, mark_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + mark_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s: torch.Tensor, mark: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, mark], dim=0)
        return self.net(x)


class ValueNN(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class GaussianPolicy(nn.Module):
    """
    Continuous bounded actions in [0,1]^action_dim via tanh + affine.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, log_std_min: float = -4.0, log_std_max: float = 1.0):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * action_dim),
        )

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(s)
        mu = out[: self.action_dim]
        log_std = out[self.action_dim :]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(s)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        # Use non-reparameterized sampling so policy-gradient uses a score-function estimator
        # (avoids unintended pathwise gradients through the sampled action).
        z = dist.sample()
        # tanh-squash -> (-1,1), then map to (0,1)
        u = torch.tanh(z)
        a = 0.5 * (u + 1.0)
        # log_prob with tanh correction (optional). Here used for diagnostics only.
        log_prob = dist.log_prob(z).sum()
        # correction: log |det da/dz| = log(0.5 * (1 - tanh(z)^2))
        corr = torch.log(0.5 * (1.0 - u.pow(2)) + 1e-12).sum()
        log_prob = log_prob - corr
        return a, log_prob

    def mode(self, s: torch.Tensor) -> torch.Tensor:
        mu, _ = self(s)
        a = 0.5 * (torch.tanh(mu) + 1.0)
        return a


def rk4_step(f, s: torch.Tensor, a: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    k1 = f(s, a, t)
    k2 = f(s + 0.5 * dt * k1, a, t + 0.5 * dt)
    k3 = f(s + 0.5 * dt * k2, a, t + 0.5 * dt)
    k4 = f(s + dt * k3, a, t + dt)
    return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_rk4(f, s0: torch.Tensor, a: torch.Tensor, t0: float, dt: float, n_steps: int = 4) -> torch.Tensor:
    """
    Integrate ds/dt = f(s,a,t) from t0 over dt using RK4 with n_steps sub-steps.
    """
    if dt <= 0:
        return s0
    n_steps = max(1, int(n_steps))
    h = dt / n_steps
    s = s0
    t = t0
    for _ in range(n_steps):
        s = rk4_step(f, s, a, t, h)
        t += h
    return s


###############################################################################
#                            7. Metrics & reward                               #
###############################################################################

@dataclass
class WirelessParams:
    P_tx_dBm: float = -10.0
    PL0_dB: float = 40.0
    n_pathloss: float = 2.2
    d0: float = 1.0
    noise_std_dB: float = 1.0
    jam_dB: float = 20.0  # max additional attenuation at jam_strength=1


def compute_rssi_dBm(pos: np.ndarray, edges: List[Tuple[int,int,float]], jam_strength: float, wp: WirelessParams) -> float:
    """
    Compute mean RSSI over existing edges (undirected). Weighted edges are allowed.
    """
    if len(edges) == 0:
        # No communication: treat as very poor quality
        return -150.0
    vals = []
    for i, j, w in edges:
        if w <= 0.0:
            continue
        dx = pos[i] - pos[j]
        d = float(np.sqrt((dx*dx).sum()) + 1e-6)
        PL = wp.PL0_dB + 10.0 * wp.n_pathloss * math.log10(d / wp.d0 + 1e-12)
        jam = wp.jam_dB * float(jam_strength)
        noise = np.random.randn() * wp.noise_std_dB
        rssi = wp.P_tx_dBm - PL - jam + noise
        vals.append(rssi)
    if len(vals) == 0:
        return -150.0
    return float(np.mean(vals))


def hub_dominance(deg: np.ndarray, top_frac: float = 0.1) -> float:
    s = float(deg.sum())
    if s <= 0.0:
        return 0.0
    k = max(1, int(round(top_frac * deg.size)))
    idx = topk_indices(deg, k)
    return float(deg[idx].sum() / s)


def percolation_threshold_zeta(deg: np.ndarray) -> float:
    """
    ζ_c = <k> / (<k^2> - <k> + eps)
    """
    k_mean = float(deg.mean())
    k2_mean = float((deg*deg).mean())
    denom = (k2_mean - k_mean)
    if denom <= 1e-12:
        return 1.0  # extremely fragile / tree-like
    z = k_mean / denom
    # Keep in [0,1] for stability
    return float(max(0.0, min(1.0, z)))


def giant_component_fraction(graph: AttackGraph) -> float:
    """
    O(N+|E|) BFS for largest connected component size / N.
    """
    N = graph.N
    seen = np.zeros(N, dtype=np.bool_)
    best = 0
    for s in range(N):
        if seen[s]:
            continue
        # BFS/DFS
        stack = [s]
        seen[s] = True
        cnt = 0
        while stack:
            u = stack.pop()
            cnt += 1
            for v in graph.neigh[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        best = max(best, cnt)
    return float(best / max(1, N))


###############################################################################
#                        8. Optional exact DeepPER                              #
###############################################################################

class DenseGCN(nn.Module):
    """
    Dense GCN used only for exact DeepPER diagnostic.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        assert num_layers >= 1
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()

    @staticmethod
    def normalize_dense(A: torch.Tensor) -> torch.Tensor:
        """
        A: (B,N,N) or (N,N)
        returns D^{-1/2} (A+I) D^{-1/2}
        """
        if A.dim() == 2:
            A = A.unsqueeze(0)
        B, N, _ = A.shape
        I = torch.eye(N, dtype=A.dtype, device=A.device).unsqueeze(0).expand(B, N, N)
        A_hat = A + I
        deg = A_hat.sum(dim=-1)  # (B,N)
        d_inv_sqrt = 1.0 / torch.sqrt(deg + 1e-12)
        D1 = d_inv_sqrt.unsqueeze(-1)  # (B,N,1)
        D2 = d_inv_sqrt.unsqueeze(-2)  # (B,1,N)
        return D1 * A_hat * D2

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        X: (B,N,F)
        A_norm: (B,N,N)
        """
        h = X
        for li, lin in enumerate(self.layers):
            h = torch.matmul(A_norm, h)
            h = lin(h)
            if li != len(self.layers) - 1:
                h = self.act(h)
        return h


def compute_deepper_vectorized(
    A: torch.Tensor,
    X: torch.Tensor,
    gcn: DenseGCN,
    chunk: int = 32,
) -> torch.Tensor:
    """
    Exact vectorized DeepPER as described by the paper's vectorization section:
    - For each i, remove i by zeroing i-th row/col in A (incident edges removed).
    - Zero out node i features.
    - Run batched dense GCN to get updated embeddings H'(i).
    - Compute cosine similarity between original H and updated H'(i), average over nodes.
    Return DeepPER_i = 1 - mean_j cos(H_j, H'_j(i)).

    A: (N,N) dense adjacency in {0,1} (or weights)
    X: (N,F)
    Returns: (N,) torch tensor
    """
    device = X.device
    N = A.size(0)

    # Original embeddings
    A0 = DenseGCN.normalize_dense(A).squeeze(0)
    H0 = gcn(X.unsqueeze(0), A0.unsqueeze(0)).squeeze(0)  # (N,D)

    # Chunked batching to reduce memory
    deepper = torch.zeros(N, dtype=torch.float32, device=device)
    onesN = torch.ones((N,), dtype=torch.float32, device=device)

    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        B = end - start

        A_batch = A.unsqueeze(0).repeat(B, 1, 1)  # (B,N,N)
        X_batch = X.unsqueeze(0).repeat(B, 1, 1)  # (B,N,F)

        idx = torch.arange(start, end, device=device)

        # Remove node i: zero row/col
        A_batch[torch.arange(B, device=device), idx, :] = 0.0
        A_batch[torch.arange(B, device=device), :, idx] = 0.0

        # Zero node feature row
        X_batch[torch.arange(B, device=device), idx, :] = 0.0

        A_norm = DenseGCN.normalize_dense(A_batch)  # (B,N,N)
        H_new = gcn(X_batch, A_norm)  # (B,N,D)

        # Cosine similarity per node j
        H0_norm = H0 / (H0.norm(dim=-1, keepdim=True) + 1e-12)
        Hn_norm = H_new / (H_new.norm(dim=-1, keepdim=True) + 1e-12)
        cos = (Hn_norm * H0_norm.unsqueeze(0)).sum(dim=-1)  # (B,N)
        cos_mean = cos.mean(dim=-1)  # (B,)
        deepper[idx] = 1.0 - cos_mean

    return deepper


###############################################################################
#                         9. ACCA full model & env                              #
###############################################################################

@dataclass
class ACCAConfig:
    N: int = 50
    F: int = 16          # raw node feature dim
    H: int = 16          # GCN embedding dim
    gcn_hidden: int = 32
    gcn_layers: int = 2

    # Initial graph (Erdos-Renyi)
    p_init: float = 0.05
    mean_degree_init: float = -1.0  # if >0, overrides p_init via p = mean_degree/(N-1)

    # DeepPA
    alpha0: float = 0.6
    Delta_alpha: float = 0.002
    alpha_noise_std: float = 0.05
    superpos_states: int = 3
    superpos_delta: float = 0.15

    # Degree update
    deltaK: int = 4

    # DeepPER / edge
    xi_d: float = 0.35
    lambda0_edge: float = 0.4

    # Continuous-time RL
    action_dim: int = 3  # [isolate, jam, misinfo]
    discount_rate: float = 0.15   # gamma in exp(-gamma t)
    dt: float = 1.0
    rk4_steps: int = 4

    # Reward coefficients (γ1,γ2,γ3 in Eq (3))
    rew_gamma1_L: float = 1.0
    rew_gamma2_PA: float = 0.5
    rew_gamma3_PER: float = 0.5

    # Defense action strengths
    isolate_max_frac: float = 0.1   # max fraction of nodes to isolate at action=1
    misinfo_sigma: float = 0.10     # feature noise std at action=1
    jam_scale: float = 0.6          # how much jamming attenuates GCN edge weights

    # Event sampling
    max_events_per_step: int = 3
    nonedge_candidates: int = 200   # candidate non-edges for creation jumps (linear cap)

    # Edge recovery / stochastic reconnection (DeepPA stochastic connections proxy)
    # After defensive removals and/or stochastic dissolutions, a fraction of removed edges
    # re-appear each step to preserve network volatility.
    edge_restore_frac_low: float = 0.10   # restore at least ~10% of removed-edge pool
    edge_restore_frac_high: float = 0.20  # restore at most ~20% of removed-edge pool
    edge_restore_max_per_step: int = -1   # if >0, cap number of restored edges per step

    # Optimization
    lr: float = 1e-3
    grad_clip: float = 1.0
    device: str = "cpu"

    # Diagnostics
    deepper_exact: bool = False     # compute exact DeepPER (small N only)

    # Seeds
    seed: int = 7


class ACCAAgent:
    """
    Holds all neural networks and optimizers.
    """
    def __init__(self, cfg: ACCAConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # State readout is mean+max pooling -> 2H
        self.state_dim = 2 * cfg.H
        self.mark_dim = 6  # [type_onehot(2), i_norm, j_norm, dk_norm, padding]

        self.gcn = SparseGCN(cfg.F, cfg.gcn_hidden, cfg.H, num_layers=cfg.gcn_layers).to(self.device)
        self.node_int = NodeIntensityNN(cfg.H).to(self.device)
        self.edge_int = EdgeIntensityNN(cfg.H).to(self.device)
        self.dk_net = DegreeDeltaCategorical(cfg.H, K=cfg.deltaK).to(self.device)

        self.entangle = EntanglementNN(self.state_dim, S=cfg.superpos_states).to(self.device)

        self.dynamics = DynamicsNN(self.state_dim, cfg.action_dim).to(self.device)
        self.jump_map = JumpMapNN(self.state_dim, self.mark_dim).to(self.device)

        self.value = ValueNN(self.state_dim).to(self.device)
        self.policy = GaussianPolicy(self.state_dim, cfg.action_dim).to(self.device)

        # Optimizers
        params_sn = list(self.gcn.parameters()) + list(self.node_int.parameters()) + list(self.edge_int.parameters()) + list(self.dk_net.parameters()) + list(self.entangle.parameters())
        self.opt_sn = optim.Adam(params_sn, lr=cfg.lr)

        params_ctrl = list(self.value.parameters()) + list(self.policy.parameters()) + list(self.dynamics.parameters()) + list(self.jump_map.parameters())
        self.opt_ctrl = optim.Adam(params_ctrl, lr=cfg.lr)

    def graph_embed(self, graph: AttackGraph, X: torch.Tensor, jam_strength: float) -> torch.Tensor:
        edge_index, edge_weight = build_gcn_edges(graph, jam_strength=jam_strength, jam_scale=self.cfg.jam_scale)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        return self.gcn(X, edge_index, edge_weight)

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        # H: (N,H) -> (2H,)
        mean = H.mean(dim=0)
        mx = H.max(dim=0).values
        return torch.cat([mean, mx], dim=0)

    def alpha_states(self, t: float) -> np.ndarray:
        """
        α(t)=α0 + Δα t + noise, then superposition with offsets ±superpos_delta.
        Returns alpha_states (S,) as numpy.
        """
        base = self.cfg.alpha0 + self.cfg.Delta_alpha * float(t) + np.random.randn() * self.cfg.alpha_noise_std
        S = self.cfg.superpos_states
        if S == 1:
            return np.array([base], dtype=np.float64)
        # symmetric offsets
        offsets = np.linspace(-self.cfg.superpos_delta, self.cfg.superpos_delta, S, dtype=np.float64)
        return base + offsets


class ACCAEnv:
    """
    Environment that evolves an attacker network graph using SNJNG + DeepPA/DeepPER,
    while defender actions modify the graph / features.

    The environment is intended to be differentiable for likelihood (intensity) training
    through the *log-likelihood terms*, but the event sampling itself is not differentiable.
    """
    def __init__(self, cfg: ACCAConfig, agent: ACCAAgent):
        self.cfg = cfg
        self.agent = agent
        self.device = agent.device

        self.graph = AttackGraph(cfg.N)
        self.t = 0.0

        # Random initial edges (Erdos-Renyi)
        p = self._initial_edge_prob()
        for i in range(cfg.N):
            for j in range(i + 1, cfg.N):
                if np.random.rand() < p:
                    self.graph.add_edge(i, j, 1.0)

        # Node features
        self.X0 = torch.randn(cfg.N, cfg.F, dtype=torch.float32, device=self.device) * 0.5
        self.X = self.X0.clone()

        # Node positions for RSSI
        self.pos = np.random.rand(cfg.N, 2).astype(np.float32) * 50.0
        self.wp = WirelessParams()

        # last action context
        self.jam_strength = 0.0

        # Edge recovery bookkeeping: pool of edges that were removed (by defense or dynamics)
        # and can be stochastically restored in subsequent steps.
        self._removed_edges: Dict[Tuple[int, int], float] = {}
        self._removed_this_step: int = 0
        self._restored_this_step: int = 0

    def _initial_edge_prob(self) -> float:
        """Return initial Erdos–Renyi edge probability.

        If cfg.mean_degree_init > 0, use p = mean_degree_init / (N-1).
        Otherwise use cfg.p_init.
        """
        try:
            md = float(getattr(self.cfg, 'mean_degree_init', -1.0))
        except Exception:
            md = -1.0
        if md > 0.0:
            denom = max(1.0, float(self.cfg.N - 1))
            p = md / denom
        else:
            p = float(getattr(self.cfg, 'p_init', 0.05))
        # clamp to [0,1]
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        return float(p)

    def reset(self) -> torch.Tensor:
        self.t = 0.0
        # reset graph
        self.graph = AttackGraph(self.cfg.N)
        p = self._initial_edge_prob()
        for i in range(self.cfg.N):
            for j in range(i + 1, self.cfg.N):
                if np.random.rand() < p:
                    self.graph.add_edge(i, j, 1.0)
        self.graph.assert_symmetric()

        self.X0 = torch.randn(self.cfg.N, self.cfg.F, dtype=torch.float32, device=self.device) * 0.5
        self.X = self.X0.clone()
        self.pos = np.random.rand(self.cfg.N, 2).astype(np.float32) * 50.0
        self.jam_strength = 0.0

        # reset edge-recovery pool
        self._removed_edges = {}
        self._removed_this_step = 0
        self._restored_this_step = 0

        # compute initial state
        with torch.no_grad():
            H = self.agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
            s = self.agent.readout(H)
        return s

    # -------------------------------------------------------------------------
    # Edge recovery bookkeeping helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _edge_key(i: int, j: int) -> Tuple[int, int]:
        if i < j:
            return int(i), int(j)
        return int(j), int(i)

    def _remove_edge_with_record(self, i: int, j: int) -> None:
        """Remove an edge and store it in the removed-edge pool for later restoration."""
        if i == j:
            return
        if not self.graph.has_edge(int(i), int(j)):
            return
        a, b = self._edge_key(int(i), int(j))
        w = float(self.graph.edge_weight(a, b))
        if w <= 0.0:
            w = 1.0
        self._removed_edges[(a, b)] = w
        self.graph.remove_edge(a, b)
        self._removed_this_step += 1

    def _add_edge_with_discard(self, i: int, j: int, weight: float = 1.0) -> None:
        """Add an edge and, if it existed in the removed-edge pool, discard it from the pool."""
        if i == j:
            return
        a, b = self._edge_key(int(i), int(j))
        self.graph.add_edge(a, b, float(weight))
        # If this edge was previously removed, consider it "restored" by dynamics.
        if (a, b) in self._removed_edges:
            del self._removed_edges[(a, b)]

    def _flip_edge_with_record(self, i: int, j: int, weight_if_add: float = 1.0) -> None:
        """Flip an edge (toggle). If removed, record it; if added, discard from removed pool."""
        if i == j:
            return
        if self.graph.has_edge(int(i), int(j)):
            self._remove_edge_with_record(int(i), int(j))
        else:
            self._add_edge_with_discard(int(i), int(j), weight=float(weight_if_add))

    def _restore_some_removed_edges(self) -> int:
        """
        Stochastically restore a fraction of edges that were previously removed.

        This approximates the "stochastic connections" idea (Bernoulli presence of edges)
        used to preserve volatility while remaining linear-time.

        Returns:
            Number of edges actually restored into the graph.
        """
        # Allow turning off via config
        low = float(getattr(self.cfg, "edge_restore_frac_low", 0.0))
        high = float(getattr(self.cfg, "edge_restore_frac_high", 0.0))
        if high <= 0.0:
            self._restored_this_step = 0
            return 0
        if low < 0.0:
            low = 0.0
        if high < low:
            high = low

        pool_size = len(self._removed_edges)
        if pool_size == 0:
            self._restored_this_step = 0
            return 0

        # Paper-aligned stochastic repair: draw a per-edge restore probability p in [low, high],
        # then sample how many edges to restore as a Binomial(|pool|, p). This yields the intended
        # 10–20% restoration rate *in expectation* while remaining O(k) by sampling without replacement.
        p_restore = random.uniform(low, high)
        k = int(np.random.binomial(pool_size, p_restore))
        k = max(0, min(k, pool_size))

        cap = int(getattr(self.cfg, "edge_restore_max_per_step", -1))
        if cap > 0:
            k = min(k, cap)

        if k <= 0:
            self._restored_this_step = 0
            return 0

        keys = list(self._removed_edges.keys())
        chosen = random.sample(keys, k)

        restored = 0
        for (a, b) in chosen:
            w = float(self._removed_edges.get((a, b), 1.0))
            if not self.graph.has_edge(a, b):
                self.graph.add_edge(a, b, w)
                restored += 1
            # Remove from pool regardless; if it already exists, it is effectively restored.
            if (a, b) in self._removed_edges:
                del self._removed_edges[(a, b)]

        self._restored_this_step = int(restored)
        return int(restored)


    def _apply_defense(self, action: np.ndarray) -> None:
        """
        action components in [0,1]:
          a_iso: isolate top-degree nodes (remove incident edges)
          a_jam: jamming strength
          a_mis: add feature noise
        """
        assert action.shape == (self.cfg.action_dim,)
        a_iso = float(action[0])
        a_jam = float(action[1]) if self.cfg.action_dim > 1 else 0.0
        a_mis = float(action[2]) if self.cfg.action_dim > 2 else 0.0

        # isolate
        deg = self.graph.degrees()
        k_isolate = int(round(a_iso * self.cfg.isolate_max_frac * self.cfg.N))
        idx_iso = topk_indices(deg, k_isolate)
        for i in idx_iso:
            # remove all incident edges
            neigh = list(self.graph.neigh[int(i)])
            for j in neigh:
                self._remove_edge_with_record(int(i), int(j))

        # jamming (affects GCN weights & RSSI)
        self.jam_strength = max(0.0, min(1.0, a_jam))

        # misinformation: corrupt node features (continuous drift proxy)
        if a_mis > 0.0:
            noise = torch.randn_like(self.X) * (self.cfg.misinfo_sigma * a_mis)
            self.X = self.X + noise

    def _sample_non_edges(self, num: int) -> List[Tuple[int, int]]:
        """
        Sample candidate non-edges for edge creation intensity evaluation.
        Complexity: O(num) expected (rejection sampling).
        """
        N = self.cfg.N
        out = []
        tries = 0
        max_tries = max(10*num, 1000)
        while len(out) < num and tries < max_tries:
            i = random.randrange(N)
            j = random.randrange(N)
            if i == j:
                tries += 1
                continue
            if self.graph.has_edge(i, j):
                tries += 1
                continue
            a, b = (i, j) if i < j else (j, i)
            out.append((a, b))
            tries += 1
        return out

    def _event_mark(self, event_type: str, i: int, j: int, delta_k: int) -> torch.Tensor:
        """
        mark vector ξ encoding:
          [node_event, edge_event, i_norm, j_norm, dk_norm, 1]
        """
        node_flag = 1.0 if event_type == "node" else 0.0
        edge_flag = 1.0 if event_type == "edge" else 0.0
        N = max(1, self.cfg.N - 1)
        i_norm = float(i) / N
        j_norm = float(j) / N
        dk_norm = float(delta_k) / max(1, self.cfg.deltaK)
        mark = torch.tensor([node_flag, edge_flag, i_norm, j_norm, dk_norm, 1.0], dtype=torch.float32, device=self.device)
        return mark

    def _apply_node_jump(
        self,
        node_i: int,
        delta_k: int,
        t: float,
        state_s: torch.Tensor,
        *,
        targets: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Apply Θ_jump (Eq. 7) for node i using DeepPA target selection (Eq. 6,10,11).

        Returns:
            logp_targets: torch scalar (sum log-probabilities) for the selected targets.

        Notes:
            - For Δk>0, we add exactly Δk distinct edges (clamped feasible) by sampling
              targets *without replacement* from Π(k,t) restricted to non-neighbors.
              We accumulate the sequential without-replacement log-prob so that
              EntanglementNN γ_s(t) is learnable via the SNJNG NLL.
            - For Δk<0, we remove exactly -Δk edges (clamped feasible) using a PER-inspired
              proxy: remove edges to the highest-degree neighbors first (targeted deletion).
        """
        N = int(self.cfg.N)
        device = self.device

        node_i = int(node_i)
        delta_k = int(delta_k)

        logp_sum = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Feasible clamp
        deg_i = len(self.graph.neigh[node_i])
        max_add = (N - 1) - deg_i
        max_rem = deg_i
        if delta_k > 0:
            delta_k = min(delta_k, max_add)
        else:
            delta_k = -min(-delta_k, max_rem)

        if delta_k == 0:
            return logp_sum

        # Degree vector (no grad)
        deg_np = self.graph.degrees().astype(np.float32)

        # DeepPA exponent superposition (no grad; stochastic per paper Eq. 10)
        alpha_np = self.agent.alpha_states(t).astype(np.float32)

        deg_t = torch.tensor(deg_np, dtype=torch.float32, device=device)
        alpha_t = torch.tensor(alpha_np, dtype=torch.float32, device=device)

        # Entanglement weights γ_s(t) from state (learned; keep in graph)
        gamma_t = self.agent.entangle(state_s, t)  # (S,)

        # Full-node PA distribution Π(k,t)
        p_all = pa_target_distribution_torch(deg_t, alpha_t, gamma_t)  # (N,)

        if delta_k > 0:
            # Candidate targets: non-neighbors excluding self
            candidates = [j for j in range(N) if j != node_i and (not self.graph.has_edge(node_i, j))]
            if not candidates:
                return logp_sum

            # How many edges to add
            k = min(delta_k, len(candidates))
            if k <= 0:
                return logp_sum

            # Determine ordered target list
            chosen: List[int] = []
            if targets is not None:
                # Filter to feasible unique candidates (preserve order)
                seen = set()
                for j in targets:
                    jj = int(j)
                    if jj == node_i:
                        continue
                    if jj < 0 or jj >= N:
                        continue
                    if self.graph.has_edge(node_i, jj):
                        continue
                    if jj in seen:
                        continue
                    if jj not in candidates:
                        continue
                    chosen.append(jj)
                    seen.add(jj)
                    if len(chosen) >= k:
                        break
            # If not enough observed targets, sample the rest
            # We sample sequentially without replacement from the *restricted* distribution.
            cand = torch.tensor(candidates, dtype=torch.int64, device=device)
            p_cand = p_all[cand]
            p_cand = p_cand / p_cand.sum().clamp_min(1e-12)

            def _remove_at(idx_in_cand: torch.Tensor, cand_t: torch.Tensor, p_t: torch.Tensor):
                mask = torch.ones((cand_t.numel(),), dtype=torch.bool, device=device)
                mask[idx_in_cand] = False
                cand_new = cand_t[mask]
                p_new = p_t[mask]
                if p_new.numel() > 0:
                    p_new = p_new / p_new.sum().clamp_min(1e-12)
                return cand_new, p_new

            # First, score any provided targets in order
            chosen_actual: List[int] = []
            for j in chosen:
                # Locate j in current candidate tensor
                idx = (cand == int(j)).nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                idx0 = idx[0]
                dist = torch.distributions.Categorical(probs=p_cand)
                logp_sum = logp_sum + dist.log_prob(idx0)
                chosen_actual.append(int(j))
                cand, p_cand = _remove_at(idx0, cand, p_cand)
                if cand.numel() == 0:
                    break

            # Then sample remaining targets
            while (len(chosen_actual) < k) and (cand.numel() > 0):
                dist = torch.distributions.Categorical(probs=p_cand)
                idx_s = dist.sample()
                logp_sum = logp_sum + dist.log_prob(idx_s)
                j = int(cand[idx_s].item())
                chosen_actual.append(j)
                cand, p_cand = _remove_at(idx_s, cand, p_cand)

            # Add edges (exactly len(chosen_actual) <= k)
            for j in chosen_actual:
                self._add_edge_with_discard(node_i, int(j), 1.0)

            return logp_sum

        # -------------------- Δk < 0 : edge removals --------------------
        # PER-inspired targeted deletion proxy: remove edges to the highest-degree neighbors first.
        neigh = list(self.graph.neigh[node_i])
        k = min(-delta_k, len(neigh))
        if k <= 0:
            return logp_sum
        neigh_arr = np.array(neigh, dtype=np.int64)
        neigh_deg = deg_np[neigh_arr]
        # pick top-k neighbors by degree (linear-time argpartition)
        idx = topk_indices(neigh_deg.astype(np.float64), k)
        to_remove = neigh_arr[idx]
        for j in to_remove:
            self._remove_edge_with_record(node_i, int(j))
        return logp_sum

    def step(self, action_t: torch.Tensor, observed_events: Optional[List[Dict[str, object]]] = None) -> Dict[str, object]:
        """
        Execute one continuous-time interval of length dt, allowing up to max_events_per_step.

        Args:
            action_t: defender action in [0,1]^A
            observed_events: optional list of externally provided events for *supervised* SNJNG
                likelihood evaluation. Each event dict supports:
                    {"tau": float in (0,dt), "type": "node"|"edge", "i": int, "j": int (edge),
                     "delta_k": int (node), "targets": List[int] (node, optional)}
                If None, events are sampled from the current (self-generated) intensities.

        Returns dict with:
          - s, s_next (state vectors)
          - s_pred_next (NODE + jump-map predicted state, for dynamics learning)
          - reward_integral (float) : ∫_t^{t+dt} e^{-γ(u-t)} r(u) du
          - td_discount (float) : e^{-γ dt}
          - nll_node, nll_edge (torch scalars) for SNJNG likelihood
          - jumps: list of realized jump records used by ITDE
          - debug metrics
        """
        cfg = self.cfg
        agent = self.agent
        dt = float(cfg.dt)
        gamma = float(cfg.discount_rate)
        t0 = float(self.t)

        # Reset per-step bookkeeping counters
        self._removed_this_step = 0
        self._restored_this_step = 0

        # Current embeddings/state (pre-action observation)
        H0 = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
        s = agent.readout(H0)

        # Action to numpy for env operations (already in [0,1])
        action_np = action_t.detach().cpu().numpy().astype(np.float64)
        self._apply_defense(action_np)
        self.graph.assert_symmetric()

        # Predict state evolution via NODE + jump map (for dynamics learning)
        # NOTE: We treat the observation s as the CTRL state at the beginning of the interval.
        s_pred = s.detach()

        # Accumulators for ITDE reward integral and likelihood
        reward_int: float = 0.0
        time_offset: float = 0.0

        # Likelihood accumulators (torch scalars)
        nll_node = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        nll_edge = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # Jump logs for ITDE (Eq. 5): store (tau, s_pre, s_post)
        jumps: List[Dict[str, object]] = []

        # Optional exact DeepPER (diagnostic)
        deepper_val = None

        # -------------------- Event-prediction "accuracy" metrics --------------------
        type_total = 0
        type_correct = 0
        node_events = 0
        edge_events = 0
        node_top1_correct = 0
        node_hit5_count = 0
        node_hit10_count = 0
        edge_top1_correct = 0
        edge_hit5_count = 0
        edge_hit10_count = 0

        remaining = dt
        ev_count = 0
        truncated = False

        # Cache tunneling hyperparameter tensor
        xi_d_t = torch.tensor(cfg.xi_d, dtype=torch.float32, device=self.device)

        def _segment_quantities(H_seg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int,float,bool]], float, float]:
            """Compute (lam_node, lam_edge, cand_pairs, Lam_val, r_t) for current graph."""
            lam_node = agent.node_int(H_seg)  # (N,)

            # candidate edges: existing edges + sampled non-edges (linear-time approximation)
            edges_und = self.graph.undirected_edges()
            cand_pairs: List[Tuple[int,int,float,bool]] = []  # (i,j,w,is_existing)
            for i, j, w in edges_und:
                cand_pairs.append((int(i), int(j), float(w), True))
            # sample non-edges for potential creation
            non_edges = self._sample_non_edges(min(cfg.nonedge_candidates, max(0, 2 * len(edges_und) + 10)))
            for (i, j) in non_edges:
                cand_pairs.append((int(i), int(j), 1.0, False))

            lam_edge_list: List[torch.Tensor] = []
            for (i, j, _w, _existed) in cand_pairs:
                hi = H_seg[i]
                hj = H_seg[j]
                Pij = cosine_squared(hi, hj)
                Tij = tunneling_term(hi, hj, xi_d_t)
                Pp = adjusted_connection_probability(Pij, Tij).clamp(0.0, 1.0)
                base = agent.edge_int(hi, hj)
                lam_ij = (cfg.lambda0_edge * base * Tij * Pp)
                lam_edge_list.append(lam_ij)
            if lam_edge_list:
                lam_edge = torch.stack(lam_edge_list)
            else:
                lam_edge = torch.zeros((0,), dtype=torch.float32, device=self.device)

            Lam_total = lam_node.sum() + lam_edge.sum()
            Lam_val = float(Lam_total.detach().cpu().item())

            # Reward terms (computed from current graph)
            deg = self.graph.degrees()
            L_t = compute_rssi_dBm(self.pos, self.graph.undirected_edges(), jam_strength=self.jam_strength, wp=self.wp)
            PA_t = 1.0 - hub_dominance(deg, top_frac=0.1)
            PER_t = percolation_threshold_zeta(deg)
            r_t = (-cfg.rew_gamma1_L * L_t) + (cfg.rew_gamma2_PA * PA_t) + (cfg.rew_gamma3_PER * PER_t)
            return lam_node, lam_edge, cand_pairs, Lam_val, float(r_t)

        def _edge_lambda_from_embeddings(hi: torch.Tensor, hj: torch.Tensor) -> torch.Tensor:
            Pij = cosine_squared(hi, hj)
            Tij = tunneling_term(hi, hj, xi_d_t)
            Pp = adjusted_connection_probability(Pij, Tij).clamp(0.0, 1.0)
            base = agent.edge_int(hi, hj)
            return (cfg.lambda0_edge * base * Tij * Pp)

        # -------------------- 1) Supervised mode: replay externally provided events --------------------
        if observed_events is not None:
            # Sort + filter to valid window
            events: List[Dict[str, object]] = []
            for e in observed_events:
                try:
                    tau = float(e.get('tau', 0.0))
                except Exception:
                    continue
                if tau <= 0.0 or tau >= dt:
                    continue
                if 'type' not in e:
                    continue
                events.append(e)
            events.sort(key=lambda d: float(d.get('tau', 0.0)))

            for e in events:
                if remaining <= 1e-12:
                    break
                if ev_count >= cfg.max_events_per_step:
                    truncated = True
                    break

                tau = float(e.get('tau', 0.0))
                seg_dt = max(0.0, tau - time_offset)

                # Segment dynamics up to tau
                H_seg = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
                s_seg = agent.readout(H_seg)
                lam_node, lam_edge, _cand_pairs, _Lam_val, r_t = _segment_quantities(H_seg)

                # Integrate reward + likelihood over (time_offset, tau)
                if seg_dt > 0.0:
                    reward_int += math.exp(-gamma * time_offset) * analytic_discounted_integral(r_t, seg_dt, gamma)
                    nll_node = nll_node + lam_node.sum() * seg_dt
                    nll_edge = nll_edge + lam_edge.sum() * seg_dt
                    s_pred = integrate_rk4(agent.dynamics, s_pred, action_t, t0 + time_offset, seg_dt, n_steps=cfg.rk4_steps)

                    time_offset += seg_dt
                    remaining = dt - time_offset

                # Apply observed event at tau
                etype = str(e.get('type'))

                if etype == 'node':
                    node_i = int(e.get('i', 0))
                    node_i = max(0, min(cfg.N - 1, node_i))

                    # negative log intensity at event time
                    nll_node = nll_node - safe_log(lam_node[node_i])

                    # observed delta_k if provided; otherwise sample
                    if 'delta_k' in e:
                        delta_k = int(e.get('delta_k', 0))
                        # supervise dk distribution if within support
                        nll_node = nll_node - agent.dk_net.log_prob(H_seg[node_i].detach(), delta_k)
                    else:
                        delta_k, logp_dk = agent.dk_net.sample(H_seg[node_i].detach())
                        # keep this as weak regularizer in self-generated mark sampling
                        nll_node = nll_node - logp_dk * 0.01

                    # Apply Θ_jump + entanglement log-prob if targets given
                    targets = e.get('targets', None)
                    if targets is not None and not isinstance(targets, list):
                        targets = None
                    s_pre = s_seg.detach()
                    logp_targets = self._apply_node_jump(node_i, delta_k, t0 + time_offset, s_seg.detach(), targets=targets)
                    nll_node = nll_node - logp_targets

                    H_post = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
                    s_post = agent.readout(H_post).detach()
                    jumps.append({"tau": float(time_offset), "type": "node", "s_pre": s_pre, "s_post": s_post})

                    # CTRL jump-map update
                    mark = self._event_mark('node', node_i, node_i, delta_k)
                    s_pred = s_pred + agent.jump_map(s_pred, mark)

                    ev_count += 1

                elif etype == 'edge':
                    i = int(e.get('i', 0))
                    j = int(e.get('j', 0))
                    i = max(0, min(cfg.N - 1, i))
                    j = max(0, min(cfg.N - 1, j))
                    if i == j:
                        # ignore self-loop events
                        continue

                    s_pre = s_seg.detach()
                    lam_ij = _edge_lambda_from_embeddings(H_seg[i], H_seg[j])
                    nll_edge = nll_edge - safe_log(lam_ij)

                    self._flip_edge_with_record(i, j, weight_if_add=1.0)

                    H_post = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
                    s_post = agent.readout(H_post).detach()
                    jumps.append({"tau": float(time_offset), "type": "edge", "s_pre": s_pre, "s_post": s_post})

                    mark = self._event_mark('edge', i, j, 0)
                    s_pred = s_pred + agent.jump_map(s_pred, mark)

                    ev_count += 1

                else:
                    # unknown event type; ignore
                    continue

                self.graph.assert_symmetric()

            # Integrate any remaining time with no further events
            remaining = dt - time_offset
            if remaining > 1e-12:
                H_seg = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
                lam_node, lam_edge, _cand_pairs, _Lam_val, r_t = _segment_quantities(H_seg)
                reward_int += math.exp(-gamma * time_offset) * analytic_discounted_integral(r_t, remaining, gamma)
                nll_node = nll_node + lam_node.sum() * remaining
                nll_edge = nll_edge + lam_edge.sum() * remaining
                s_pred = integrate_rk4(agent.dynamics, s_pred, action_t, t0 + time_offset, remaining, n_steps=cfg.rk4_steps)
                time_offset += remaining
                remaining = 0.0

        # -------------------- 2) Self-generated mode: sample events from intensities --------------------
        else:
            while remaining > 1e-12 and ev_count < cfg.max_events_per_step:
                H_seg = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
                s_seg = agent.readout(H_seg)

                lam_node, lam_edge, cand_pairs, Lam_val, r_t = _segment_quantities(H_seg)

                # Sample next event time
                if Lam_val <= 1e-12:
                    # no events; integrate full remaining
                    seg_dt = remaining
                    reward_int += math.exp(-gamma * time_offset) * analytic_discounted_integral(r_t, seg_dt, gamma)
                    nll_node = nll_node + lam_node.sum() * seg_dt
                    nll_edge = nll_edge + lam_edge.sum() * seg_dt
                    s_pred = integrate_rk4(agent.dynamics, s_pred, action_t, t0 + time_offset, seg_dt, n_steps=cfg.rk4_steps)
                    time_offset += seg_dt
                    remaining = 0.0
                    break

                # exponential waiting time
                u = random.random()
                wait = -math.log(max(u, 1e-12)) / Lam_val
                if wait >= remaining:
                    # no event within remaining
                    seg_dt = remaining
                    reward_int += math.exp(-gamma * time_offset) * analytic_discounted_integral(r_t, seg_dt, gamma)
                    nll_node = nll_node + lam_node.sum() * seg_dt
                    nll_edge = nll_edge + lam_edge.sum() * seg_dt
                    s_pred = integrate_rk4(agent.dynamics, s_pred, action_t, t0 + time_offset, seg_dt, n_steps=cfg.rk4_steps)
                    time_offset += seg_dt
                    remaining = 0.0
                    break

                # event occurs after 'wait'
                seg_dt = wait
                reward_int += math.exp(-gamma * time_offset) * analytic_discounted_integral(r_t, seg_dt, gamma)
                nll_node = nll_node + lam_node.sum() * seg_dt
                nll_edge = nll_edge + lam_edge.sum() * seg_dt
                s_pred = integrate_rk4(agent.dynamics, s_pred, action_t, t0 + time_offset, seg_dt, n_steps=cfg.rk4_steps)

                time_offset += seg_dt
                remaining -= seg_dt

                # Choose event type & which node/edge
                lam_node_sum = float(lam_node.sum().detach().cpu().item())
                p_node = lam_node_sum / max(Lam_val, 1e-12)

                # ---------- event prediction proxies (type + top-k hits) ----------
                pred_is_node = (p_node >= 0.5)

                lam_node_np = lam_node.detach().cpu().numpy()
                node_argmax = int(lam_node_np.argmax()) if lam_node_np.size > 0 else -1
                node_top5 = set(topk_indices(lam_node_np, 5))
                node_top10 = set(topk_indices(lam_node_np, 10))

                lam_edge_np = lam_edge.detach().cpu().numpy()
                if lam_edge_np.size > 0:
                    edge_argmax = int(lam_edge_np.argmax())
                    edge_top5 = set(topk_indices(lam_edge_np, 5))
                    edge_top10 = set(topk_indices(lam_edge_np, 10))
                else:
                    edge_argmax = -1
                    edge_top5 = set()
                    edge_top10 = set()

                if random.random() < p_node and lam_node_sum > 0.0:
                    # node event
                    probs = (lam_node / lam_node.sum()).detach().cpu().numpy()
                    node_i = int(np.random.choice(np.arange(cfg.N), p=probs))

                    # event-type prediction accuracy (proxy)
                    type_total += 1
                    type_correct += int(pred_is_node)

                    # node index hit-rates (top-k over node intensities)
                    node_events += 1
                    node_top1_correct += int(node_i == node_argmax)
                    node_hit5_count += int(node_i in node_top5)
                    node_hit10_count += int(node_i in node_top10)

                    # event log term
                    nll_node = nll_node - safe_log(lam_node[node_i])

                    # sample delta_k (mark)
                    delta_k, logp_dk = agent.dk_net.sample(H_seg[node_i].detach())
                    # encourage plausible small changes (regularization via negative log prob)
                    nll_node = nll_node - logp_dk * 0.01

                    # Apply jump to graph (Eq. 7) + log-prob for entanglement (targets)
                    s_pre = s_seg.detach()
                    logp_targets = self._apply_node_jump(node_i, delta_k, t0 + time_offset, s_seg.detach())
                    nll_node = nll_node - logp_targets

                    H_post = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
                    s_post = agent.readout(H_post).detach()
                    jumps.append({"tau": float(time_offset), "type": "node", "s_pre": s_pre, "s_post": s_post})

                    # Update predicted state with jump map
                    mark = self._event_mark('node', node_i, node_i, delta_k)
                    s_pred = s_pred + agent.jump_map(s_pred, mark)

                    ev_count += 1

                else:
                    # edge event (including creation via sampled non-edges)
                    if lam_edge.numel() == 0:
                        break
                    probs = (lam_edge / lam_edge.sum()).detach().cpu().numpy()
                    eidx = int(np.random.choice(np.arange(len(cand_pairs)), p=probs))

                    # event-type prediction accuracy (proxy)
                    type_total += 1
                    type_correct += int(not pred_is_node)

                    # edge index hit-rates (top-k over candidate-edge intensities)
                    edge_events += 1
                    edge_top1_correct += int(eidx == edge_argmax)
                    edge_hit5_count += int(eidx in edge_top5)
                    edge_hit10_count += int(eidx in edge_top10)

                    i, j, _w, _existed = cand_pairs[eidx]
                    nll_edge = nll_edge - safe_log(lam_edge[eidx])

                    s_pre = s_seg.detach()
                    # flip adjacency
                    self._flip_edge_with_record(i, j, weight_if_add=1.0)

                    H_post = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
                    s_post = agent.readout(H_post).detach()
                    jumps.append({"tau": float(time_offset), "type": "edge", "s_pre": s_pre, "s_post": s_post})

                    # Update predicted state with jump map
                    mark = self._event_mark('edge', i, j, 0)
                    s_pred = s_pred + agent.jump_map(s_pred, mark)

                    ev_count += 1

                self.graph.assert_symmetric()

            # If we stopped because of max_events, integrate the remaining interval with no more events.
            if remaining > 1e-12:
                if ev_count >= cfg.max_events_per_step:
                    truncated = True
                H_seg = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
                lam_node, lam_edge, _cand_pairs, _Lam_val, r_t = _segment_quantities(H_seg)
                reward_int += math.exp(-gamma * time_offset) * analytic_discounted_integral(r_t, remaining, gamma)
                nll_node = nll_node + lam_node.sum() * remaining
                nll_edge = nll_edge + lam_edge.sum() * remaining
                s_pred = integrate_rk4(agent.dynamics, s_pred, action_t, t0 + time_offset, remaining, n_steps=cfg.rk4_steps)
                time_offset += remaining
                remaining = 0.0

        # End of interval: advance env time
        self.t = t0 + dt

        # Stochastic edge recovery: restore 10–20% of the removed-edge pool per step (configurable).
        restored_edges = self._restore_some_removed_edges()
        self.graph.assert_symmetric()

        # Next embeddings/state
        H_next = agent.graph_embed(self.graph, self.X, jam_strength=self.jam_strength)
        s_next = agent.readout(H_next)

        # Optional exact DeepPER diagnostic
        if cfg.deepper_exact and cfg.N <= 200:
            A_dense = torch.tensor(self.graph.to_dense_adjacency(), dtype=torch.float32, device=self.device)
            gcn_dense = DenseGCN(cfg.F, cfg.gcn_hidden, cfg.H, num_layers=cfg.gcn_layers).to(self.device)
            with torch.no_grad():
                for li in range(min(len(gcn_dense.layers), len(agent.gcn.layers))):
                    gcn_dense.layers[li].weight.copy_(agent.gcn.layers[li].lin.weight)
                    gcn_dense.layers[li].bias.copy_(agent.gcn.layers[li].lin.bias)
            deepper_val = compute_deepper_vectorized(A_dense, self.X, gcn_dense, chunk=32).detach()

        td_discount = math.exp(-gamma * dt)

        debug = {
            't0': t0,
            't1': float(self.t),
            'dt_covered': float(time_offset),
            'truncated': int(truncated),
            'edges': self.graph.num_edges(),
            'removed_edges': int(self._removed_this_step),
            'restored_edges': int(restored_edges),
            'removed_pool': int(len(self._removed_edges)),
            'mean_deg': float(self.graph.degrees().mean()),
            'rssi': float(compute_rssi_dBm(self.pos, self.graph.undirected_edges(), jam_strength=self.jam_strength, wp=self.wp)),
            'hub_dom': float(hub_dominance(self.graph.degrees(), top_frac=0.1)),
            'zeta': float(percolation_threshold_zeta(self.graph.degrees())),
            'gcc': float(giant_component_fraction(self.graph)),
            'events': int(ev_count),
            'node_events': int(node_events),
            'edge_events': int(edge_events),
            'type_acc': float(type_correct / type_total) if type_total > 0 else 0.0,
            'node_top1_acc': float(node_top1_correct / node_events) if node_events > 0 else 0.0,
            'node_hit5': float(node_hit5_count / node_events) if node_events > 0 else 0.0,
            'node_hit10': float(node_hit10_count / node_events) if node_events > 0 else 0.0,
            'edge_top1_acc': float(edge_top1_correct / edge_events) if edge_events > 0 else 0.0,
            'edge_hit5': float(edge_hit5_count / edge_events) if edge_events > 0 else 0.0,
            'edge_hit10': float(edge_hit10_count / edge_events) if edge_events > 0 else 0.0,
        }

        return {
            's': s,
            's_next': s_next,
            's_pred_next': s_pred,
            'action_np': action_np,
            'reward_integral': float(reward_int),
            'td_discount': float(td_discount),
            'nll_node': nll_node,
            'nll_edge': nll_edge,
            'deepper': deepper_val,
            'jumps': jumps,
            'debug': debug,
        }



###############################################################################
#                         10. Training (SNJNG + CTRL)                           #
###############################################################################


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    aa = a.detach().float()
    bb = b.detach().float()
    num = float((aa * bb).sum().cpu().item())
    den = float((aa.norm() * bb.norm()).clamp_min(eps).cpu().item())
    return float(num / den)


def train_accc(
    cfg: ACCAConfig,
    steps: int = 200,
    print_every: int = 20,
    collect_history: bool = True,
) -> Tuple[ACCAAgent, ACCAEnv, List[Dict[str, float]]]:
    """
    Train SNJNG likelihood (opt_sn) + CTRL components (opt_ctrl) for 'steps'.

    Returns:
        agent: trained ACCAAgent
        env: final environment (contains final graph/features/positions)
        history: list of per-step scalar metrics for visualization/evaluation
    """
    set_seed(cfg.seed)
    agent = ACCAAgent(cfg)
    env = ACCAEnv(cfg, agent)

    s = env.reset()

    history: List[Dict[str, float]] = []

    for t in range(1, steps + 1):
        s = s.to(agent.device)

        # Sample action from policy
        a_t, logp = agent.policy.sample(s.detach())
        a_t = torch.clamp(a_t, 0.0, 1.0)

        # Step env
        out = env.step(a_t)
        action_np = out["action_np"]

        s_next = out["s_next"].detach()
        reward_int = float(out["reward_integral"])
        disc = float(out["td_discount"])

        # -------------------- SNJNG likelihood update --------------------
        nll = out["nll_node"] + out["nll_edge"]

        agent.opt_sn.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(agent.gcn.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(agent.node_int.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(agent.edge_int.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(agent.dk_net.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(agent.entangle.parameters(), cfg.grad_clip)
        agent.opt_sn.step()

        # -------------------- CTRL ITDE critic update --------------------
        # Integral TD error with jumps (Eq. 5):
        #   δ = ∫ e^{-γ(u-t)} r(u) du + Σ_{τ_j in (t,t+Δt)} e^{-γ τ_j}(V(s^+)-V(s^-)) + e^{-γΔt}V(s_next) - V(s)
        V_s = agent.value(s.detach())
        V_next = agent.value(s_next)

        jump_val = torch.tensor(0.0, dtype=torch.float32, device=agent.device)
        for rec in out.get('jumps', []):
            tau = float(rec.get('tau', 0.0))
            disc_tau = math.exp(-cfg.discount_rate * tau)
            s_pre = rec.get('s_pre', None)
            s_post = rec.get('s_post', None)
            if isinstance(s_pre, torch.Tensor) and isinstance(s_post, torch.Tensor):
                jump_val = jump_val + float(disc_tau) * (agent.value(s_post.to(agent.device)) - agent.value(s_pre.to(agent.device)))

        delta = torch.tensor(reward_int, dtype=torch.float32, device=agent.device) + jump_val + float(disc) * V_next - V_s

        loss_value = 0.5 * delta.pow(2)

        # -------------------- dynamics learning --------------------
        s_pred_next = out['s_pred_next']
        loss_dyn = 0.1 * (s_pred_next - s_next).pow(2).mean()

        # -------------------- Actor update (policy gradient using ITDE advantage) --------------------
        adv = delta.detach()
        loss_actor = -(adv * logp)

        loss_ctrl = loss_value + loss_dyn + loss_actor

        agent.opt_ctrl.zero_grad()
        loss_ctrl.backward()
        torch.nn.utils.clip_grad_norm_(agent.value.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(agent.dynamics.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(agent.jump_map.parameters(), cfg.grad_clip)
        agent.opt_ctrl.step()

        itde_error = float(delta.detach().cpu().item())
        td_error = itde_error

        # -------------------- metrics --------------------
        if collect_history:
            dbg = out["debug"]
            state_mse = float((s_pred_next.detach() - s_next.detach()).pow(2).mean().cpu().item())
            state_cos = _cosine_sim(s_pred_next, s_next)

            history.append({
                "step": float(t),
                "env_t": float(dbg.get("t1", dbg.get("t", 0.0))),
                "reward_int": float(reward_int),
                "nll": float(nll.detach().cpu().item()),
                "loss_value": float(loss_value.detach().cpu().item()),
                "loss_dyn": float(loss_dyn.detach().cpu().item()),
                "loss_actor": float(loss_actor.detach().cpu().item()),
                "loss_ctrl": float(loss_ctrl.detach().cpu().item()),
                "td_error": float(td_error),
                "state_mse": float(state_mse),
                "state_cos": float(state_cos),
                "edges": float(dbg["edges"]),
                "removed_edges": float(dbg.get("removed_edges", 0.0)),
                "restored_edges": float(dbg.get("restored_edges", 0.0)),
                "removed_pool": float(dbg.get("removed_pool", 0.0)),
                "mean_deg": float(dbg["mean_deg"]),
                "rssi": float(dbg["rssi"]),
                "hub_dom": float(dbg["hub_dom"]),
                "zeta": float(dbg["zeta"]),
                "gcc": float(dbg["gcc"]),
                "events": float(dbg["events"]),
                "type_acc": float(dbg.get("type_acc", 0.0)),
                "node_top1_acc": float(dbg.get("node_top1_acc", 0.0)),
                "node_hit5": float(dbg.get("node_hit5", 0.0)),
                "node_hit10": float(dbg.get("node_hit10", 0.0)),
                "edge_top1_acc": float(dbg.get("edge_top1_acc", 0.0)),
                "edge_hit5": float(dbg.get("edge_hit5", 0.0)),
                "edge_hit10": float(dbg.get("edge_hit10", 0.0)),
                "node_events": float(dbg.get("node_events", 0.0)),
                "edge_events": float(dbg.get("edge_events", 0.0)),
                "a_iso": float(action_np[0]) if action_np.size > 0 else 0.0,
                "a_jam": float(action_np[1]) if action_np.size > 1 else 0.0,
                "a_mis": float(action_np[2]) if action_np.size > 2 else 0.0,
            })

        s = s_next

        if (print_every > 0) and (t % print_every == 0):
            dbg = out["debug"]
            print(
                f"[t={t:4d}] nll={float(nll.detach().cpu().item()):8.3f} "
                f"td_err={td_error:8.3f} Rint={reward_int:8.3f} edges={dbg['edges']:4d} "
                f"mean_deg={dbg['mean_deg']:.2f} removed={int(dbg.get('removed_edges', 0)):4d} "
                f"restored={int(dbg.get('restored_edges', 0)):4d} pool={int(dbg.get('removed_pool', 0)):6d} "
                f"iso={float(action_np[0]) if action_np.size > 0 else 0.0:.2f} "
                f"zeta={dbg['zeta']:.3f} gcc={dbg['gcc']:.2f} events={dbg['events']}"
            )

    return agent, env, history


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.copy()
    window = min(window, x.size)
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[window:] - c[:-window]) / window
    # pad to length
    pad = np.full(window - 1, ma[0], dtype=np.float64)
    return np.concatenate([pad, ma])


def save_history_csv(history: List[Dict[str, float]], path: str) -> None:
    import csv
    if not history:
        return
    keys = list(history[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k, float("nan")) for k in keys})


def summarize_history(history: List[Dict[str, float]], window: int = 50) -> Dict[str, float]:
    if not history:
        return {}
    window = max(1, min(window, len(history)))
    last = history[-window:]
    def avg(key: str) -> float:
        vals = [float(r[key]) for r in last if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "avg_reward_int": avg("reward_int"),
        "avg_nll": avg("nll"),
        "avg_td_abs": float(np.mean([abs(float(r["td_error"])) for r in last])),
        "avg_state_mse": avg("state_mse"),
        "avg_state_cos": avg("state_cos"),
        "avg_edges": avg("edges"),
        "avg_removed_edges": avg("removed_edges"),
        "avg_restored_edges": avg("restored_edges"),
        "avg_removed_pool": avg("removed_pool"),
        "avg_rssi": avg("rssi"),
        "avg_hub_dom": avg("hub_dom"),
        "avg_zeta": avg("zeta"),
        "avg_gcc": avg("gcc"),
        "avg_events": avg("events"),
        "avg_type_acc": avg("type_acc"),
        "avg_node_top1_acc": avg("node_top1_acc"),
        "avg_edge_top1_acc": avg("edge_top1_acc"),
        "avg_node_hit10": avg("node_hit10"),
        "avg_edge_hit10": avg("edge_hit10"),
    }
    return summary


def evaluate_policy(
    agent: ACCAAgent,
    cfg: ACCAConfig,
    steps: int = 200,
    policy: str = "mode",  # "mode" or "random"
    seed: int = 999,
) -> Dict[str, float]:
    """
    Quick policy evaluation on a fresh environment (no learning).
    Returns average metrics over steps.
    """
    set_seed(seed)
    env = ACCAEnv(cfg, agent)
    s = env.reset()

    rewards = []
    rssi = []
    zeta = []
    gcc = []
    edges = []
    type_acc = []
    node_hit10 = []
    edge_hit10 = []

    with torch.no_grad():
        for _ in range(steps):
            s = s.to(agent.device)
            if policy == "random":
                a_np = np.random.rand(cfg.action_dim).astype(np.float32)
                a = torch.tensor(a_np, dtype=torch.float32, device=agent.device)
            else:
                a = agent.policy.mode(s)
                a = torch.clamp(a, 0.0, 1.0)

            out = env.step(a)
            rewards.append(float(out["reward_integral"]))
            dbg = out["debug"]
            rssi.append(float(dbg["rssi"]))
            zeta.append(float(dbg["zeta"]))
            gcc.append(float(dbg["gcc"]))
            edges.append(float(dbg["edges"]))
            type_acc.append(float(dbg.get("type_acc", 0.0)))
            node_hit10.append(float(dbg.get("node_hit10", 0.0)))
            edge_hit10.append(float(dbg.get("edge_hit10", 0.0)))
            s = out["s_next"].detach()

    return {
        "policy": policy,
        "avg_reward_int": float(np.mean(rewards)) if rewards else float("nan"),
        "avg_rssi": float(np.mean(rssi)) if rssi else float("nan"),
        "avg_zeta": float(np.mean(zeta)) if zeta else float("nan"),
        "avg_gcc": float(np.mean(gcc)) if gcc else float("nan"),
        "avg_edges": float(np.mean(edges)) if edges else float("nan"),
        "avg_type_acc": float(np.mean(type_acc)) if type_acc else float("nan"),
        "avg_node_hit10": float(np.mean(node_hit10)) if node_hit10 else float("nan"),
        "avg_edge_hit10": float(np.mean(edge_hit10)) if edge_hit10 else float("nan"),
    }


def _plot_series(x: np.ndarray, y: np.ndarray, title: str, ylabel: str, path: str, show: bool = False) -> None:
    if plt is None:
        print("matplotlib is not available; skipping plots.")
        return
    import matplotlib.pyplot as _plt  # ensure local alias
    _plt.figure(figsize=(10, 4))
    _plt.plot(x, y)
    _plt.title(title)
    _plt.xlabel("step")
    _plt.ylabel(ylabel)
    _plt.grid(True, alpha=0.3)
    _plt.tight_layout()
    _plt.savefig(path, dpi=160)
    if show:
        _plt.show()
    _plt.close()


def plot_history(history: List[Dict[str, float]], outdir: str, show: bool = False) -> None:
    import os
    if not history:
        print("No history to plot.")
        return
    os.makedirs(outdir, exist_ok=True)

    steps = np.array([h["step"] for h in history], dtype=np.float64)

    def arr(key: str) -> np.ndarray:
        return np.array([h[key] for h in history], dtype=np.float64)

    # Reward
    r = arr("reward_int")
    _plot_series(steps, r, "Reward integral per step", "reward_integral", os.path.join(outdir, "reward.png"), show=show)
    _plot_series(steps, _moving_average(r, window=max(5, len(r)//20)), "Reward (moving average)", "reward_integral (MA)", os.path.join(outdir, "reward_ma.png"), show=show)

    # Losses
    _plot_series(steps, arr("nll"), "SNJNG negative log-likelihood", "nll", os.path.join(outdir, "nll.png"), show=show)
    _plot_series(steps, arr("loss_value"), "Critic loss (MSE)", "loss_value", os.path.join(outdir, "loss_value.png"), show=show)
    _plot_series(steps, arr("loss_dyn"), "Dynamics loss (state prediction)", "loss_dyn", os.path.join(outdir, "loss_dyn.png"), show=show)
    _plot_series(steps, arr("loss_actor"), "Actor objective (negative Hamiltonian)", "loss_actor", os.path.join(outdir, "loss_actor.png"), show=show)

    # Prediction "accuracy"
    _plot_series(steps, arr("state_cos"), "State prediction cosine similarity (accuracy proxy)", "cosine(s_pred, s_next)", os.path.join(outdir, "state_cos.png"), show=show)
    _plot_series(steps, arr("state_mse"), "State prediction MSE (lower is better)", "mse", os.path.join(outdir, "state_mse.png"), show=show)

    # Event "accuracy" proxies
    if "type_acc" in history[0]:
        _plot_series(steps, arr("type_acc"), "Event type prediction accuracy (proxy)", "type_acc", os.path.join(outdir, "event_type_acc.png"), show=show)
        _plot_series(steps, arr("node_hit10"), "Node event hit@10 (proxy)", "hit@10", os.path.join(outdir, "node_hit10.png"), show=show)
        _plot_series(steps, arr("edge_hit10"), "Edge event hit@10 (proxy)", "hit@10", os.path.join(outdir, "edge_hit10.png"), show=show)

    # Graph metrics
    _plot_series(steps, arr("edges"), "Number of edges", "edges", os.path.join(outdir, "edges.png"), show=show)

    # Edge recovery diagnostics (if present)
    if "restored_edges" in history[0]:
        _plot_series(steps, arr("restored_edges"), "Restored edges per step", "restored_edges", os.path.join(outdir, "restored_edges.png"), show=show)
    if "removed_pool" in history[0]:
        _plot_series(steps, arr("removed_pool"), "Removed-edge pool size", "removed_pool", os.path.join(outdir, "removed_pool.png"), show=show)
    _plot_series(steps, arr("gcc"), "Giant component fraction", "gcc", os.path.join(outdir, "gcc.png"), show=show)
    _plot_series(steps, arr("zeta"), "Percolation threshold (zeta)", "zeta", os.path.join(outdir, "zeta.png"), show=show)
    _plot_series(steps, arr("hub_dom"), "Hub dominance (top 10% degree share)", "hub_dom", os.path.join(outdir, "hub_dom.png"), show=show)
    _plot_series(steps, arr("rssi"), "Mean RSSI (dBm)", "rssi_dBm", os.path.join(outdir, "rssi.png"), show=show)

    # Actions
    _plot_series(steps, arr("a_iso"), "Action: isolate", "a_iso", os.path.join(outdir, "action_isolate.png"), show=show)
    if "a_jam" in history[0]:
        _plot_series(steps, arr("a_jam"), "Action: jam", "a_jam", os.path.join(outdir, "action_jam.png"), show=show)
    if "a_mis" in history[0]:
        _plot_series(steps, arr("a_mis"), "Action: misinfo", "a_mis", os.path.join(outdir, "action_misinfo.png"), show=show)

    # Save CSV
    save_history_csv(history, os.path.join(outdir, "metrics.csv"))
    print(f"Saved plots + metrics.csv to: {outdir}")


def plot_graph_snapshot(graph: AttackGraph, pos: np.ndarray, path: str, title: str = "Graph snapshot", show: bool = False) -> None:
    """
    Lightweight visualization without networkx: scatter nodes + draw edges.
    """
    if plt is None:
        print("matplotlib is not available; skipping graph plot.")
        return
    import matplotlib.pyplot as _plt
    _plt.figure(figsize=(6, 6))
    _plt.scatter(pos[:, 0], pos[:, 1], s=25)
    # edges
    for (i, j), w in graph.w.items():
        if w <= 0:
            continue
        _plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], linewidth=0.7, alpha=0.5)
    _plt.title(title)
    _plt.axis("equal")
    _plt.tight_layout()
    _plt.savefig(path, dpi=160)
    if show:
        _plt.show()
    _plt.close()


###############################################################################
#                               11. Unit tests                                 #
###############################################################################

def run_unit_tests() -> None:
    print("Running unit tests...")

    # Base config for correctness tests
    cfg = ACCAConfig(
        N=20,
        F=8,
        H=8,
        gcn_hidden=16,
        gcn_layers=2,
        deepper_exact=True,
        device="cpu",
        seed=123,
    )
    set_seed(cfg.seed)
    agent = ACCAAgent(cfg)
    env = ACCAEnv(cfg, agent)
    s = env.reset()

    # ----------------------------------------------------------------------
    # 1) Intensity positivity + edge symmetry
    # ----------------------------------------------------------------------
    with torch.no_grad():
        H = agent.graph_embed(env.graph, env.X, jam_strength=0.0)
        lam_n = agent.node_int(H)
        assert torch.all(lam_n > 0), "Node intensities must be positive"

        i, j = 0, 1
        lam_e_ij = agent.edge_int(H[i], H[j])
        lam_e_ji = agent.edge_int(H[j], H[i])
        assert lam_e_ij.item() > 0 and lam_e_ji.item() > 0, "Edge intensity base must be positive"
        assert torch.allclose(lam_e_ij, lam_e_ji, atol=1e-6), "Edge intensity must be symmetric for undirected graphs"

    # ----------------------------------------------------------------------
    # 2) DeepPER adjusted probability is in [0,1]
    # ----------------------------------------------------------------------
    with torch.no_grad():
        Pij = cosine_squared(H[0], H[1])
        Tij = tunneling_term(H[0], H[1], torch.tensor(cfg.xi_d))
        Pp = adjusted_connection_probability(Pij, Tij)
        assert 0.0 <= float(Pp.item()) <= 1.0 + 1e-6, "Adjusted connection probability must be within [0,1]"

    # ----------------------------------------------------------------------
    # 3) step(): symmetry, degree consistency, and full-interval accounting
    # ----------------------------------------------------------------------
    a = torch.tensor([1.0, 0.5, 0.5], dtype=torch.float32)  # strong defense
    out = env.step(a)
    env.graph.assert_symmetric()
    deg = env.graph.degrees()
    for ii in range(env.graph.N):
        assert deg[ii] == len(env.graph.neigh[ii]), "Degree mismatch"
    assert abs(float(out["debug"].get("dt_covered", 0.0)) - float(cfg.dt)) < 1e-6, "dt_covered must equal dt"

    # ----------------------------------------------------------------------
    # 4) ITDE pieces finite + jump log structure
    # ----------------------------------------------------------------------
    assert math.isfinite(out["reward_integral"]), "Reward integral must be finite"
    assert math.isfinite(out["td_discount"]), "Discount must be finite"
    with torch.no_grad():
        v1 = agent.value(out["s"].detach())
        v2 = agent.value(out["s_next"].detach())
        assert torch.isfinite(v1).all() and torch.isfinite(v2).all(), "Value must be finite"

    jumps = out.get("jumps", [])
    assert isinstance(jumps, list), "jumps must be a list"
    for rec in jumps:
        assert "tau" in rec and "s_pre" in rec and "s_post" in rec, "jump record missing required fields"
        tau = float(rec["tau"])
        assert 0.0 <= tau <= float(cfg.dt) + 1e-6, "jump tau must be within [0,dt]"
        assert isinstance(rec["s_pre"], torch.Tensor) and isinstance(rec["s_post"], torch.Tensor), "jump states must be tensors"

    # ----------------------------------------------------------------------
    # 4b) Event accuracy proxies are in [0,1]
    # ----------------------------------------------------------------------
    dbg = out["debug"]
    for k in ["type_acc", "node_top1_acc", "node_hit5", "node_hit10", "edge_top1_acc", "edge_hit5", "edge_hit10"]:
        v = float(dbg.get(k, 0.0))
        assert 0.0 <= v <= 1.0 + 1e-6, f"{k} must be within [0,1]"

    # ----------------------------------------------------------------------
    # 5) Action affects reward (counterfactual) — requires deep copying
    # ----------------------------------------------------------------------
    env2 = ACCAEnv(cfg, agent)
    env2.reset()

    def _copy_env_state(dst: ACCAEnv, src: ACCAEnv) -> None:
        dst.graph = src.graph.clone()
        dst.X = src.X.clone()
        dst.pos = src.pos.copy()
        dst.t = float(src.t)
        dst.jam_strength = float(src.jam_strength)

    _copy_env_state(env2, env)
    out_nojam = env2.step(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))

    _copy_env_state(env2, env)
    out_jam = env2.step(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))

    assert out_nojam["reward_integral"] != out_jam["reward_integral"], "Changing action must change reward integral"

    # ----------------------------------------------------------------------
    # 6) Exact DeepPER runs (small N)
    # ----------------------------------------------------------------------
    deepper = out["deepper"]
    assert deepper is None or (deepper.shape[0] == cfg.N), "DeepPER exact must return N values"

    # ----------------------------------------------------------------------
    # 7) DeepPA entanglement gradients flow through target log-prob
    # ----------------------------------------------------------------------
    env3 = ACCAEnv(cfg, agent)
    env3.reset()
    with torch.no_grad():
        H3 = agent.graph_embed(env3.graph, env3.X, jam_strength=0.0)
        s3 = agent.readout(H3).detach()

    agent.opt_sn.zero_grad()
    logp_targets = env3._apply_node_jump(node_i=0, delta_k=3, t=float(env3.t), state_s=s3)
    loss_pa = (-logp_targets)
    loss_pa.backward()
    ent_grad = 0.0
    for p in agent.entangle.parameters():
        if p.grad is not None:
            ent_grad += float(p.grad.abs().sum().item())
    assert ent_grad > 0.0, "EntanglementNN must receive non-zero gradient from DeepPA target log-prob"

    # ----------------------------------------------------------------------
    # 8) Policy log-prob depends on parameters (policy-gradient path exists)
    # ----------------------------------------------------------------------
    agent.opt_ctrl.zero_grad()
    a_test, logp_test = agent.policy.sample(s.detach())
    loss_pg = (-logp_test)
    loss_pg.backward()
    pol_grad = 0.0
    for p in agent.policy.parameters():
        if p.grad is not None:
            pol_grad += float(p.grad.abs().sum().item())
    assert pol_grad > 0.0, "Policy parameters must receive gradient through log-prob"

    # ----------------------------------------------------------------------
    # 9) Observed-events mode runs and remains finite
    # ----------------------------------------------------------------------
    env4 = ACCAEnv(cfg, agent)
    env4.reset()
    obs = [{"tau": float(cfg.dt) * 0.5, "type": "edge", "i": 0, "j": 1}]
    out_obs = env4.step(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32), observed_events=obs)
    assert math.isfinite(out_obs["reward_integral"]), "Observed-events mode reward integral must be finite"
    assert abs(float(out_obs["debug"].get("dt_covered", 0.0)) - float(cfg.dt)) < 1e-6, "Observed-events mode must cover dt"

    # ----------------------------------------------------------------------
    # 10) Truncation path: even with max_events_per_step=1, we still cover dt
    # ----------------------------------------------------------------------
    cfgT = ACCAConfig(
        N=30,
        F=8,
        H=8,
        gcn_hidden=16,
        gcn_layers=2,
        dt=1.0,
        max_events_per_step=1,
        lambda0_edge=5.0,
        deepper_exact=False,
        device="cpu",
        seed=321,
    )
    set_seed(cfgT.seed)
    agentT = ACCAAgent(cfgT)
    envT = ACCAEnv(cfgT, agentT)
    envT.reset()
    outT = envT.step(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
    assert abs(float(outT["debug"].get("dt_covered", 0.0)) - float(cfgT.dt)) < 1e-6, "Truncated run must still cover dt"
    assert int(outT["debug"].get("events", 0)) <= int(cfgT.max_events_per_step), "events must respect max_events_per_step"

    print("All unit tests passed.")



###############################################################################
#                                    main                                     #
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tests", action="store_true", help="Run unit tests (default if no flags).")
    ap.add_argument("--demo", action="store_true", help="Run a short training/demo loop.")
    ap.add_argument("--steps", type=int, default=200, help="Number of training steps for --demo.")
    ap.add_argument("--plot", action="store_true", help="After --demo, save plots + metrics.csv (requires matplotlib).")
    ap.add_argument("--outdir", type=str, default="runs", help="Output folder for plots/metrics.")
    ap.add_argument("--show_plots", action="store_true", help="Show plots interactively (PyCharm-friendly).")
    ap.add_argument("--eval_steps", type=int, default=200, help="Evaluation steps after training (mode vs random).")

    # -------------------- Experiment / model hyperparameters (override ACCAConfig defaults) --------------------
    defaults = ACCAConfig()
    ap.add_argument("--N", type=int, default=defaults.N, help="Number of nodes in the graph.")
    ap.add_argument("--F", type=int, default=defaults.F, help="Raw node feature dimension.")
    ap.add_argument("--H", type=int, default=defaults.H, help="GCN output embedding dimension (RL state dim = 2*H).")
    ap.add_argument("--gcn_hidden", type=int, default=defaults.gcn_hidden, help="GCN hidden dimension.")
    ap.add_argument("--gcn_layers", type=int, default=defaults.gcn_layers, help="Number of GCN layers.")

    # Initial graph density: choose ONE of the following (mean_degree overrides p_init)
    ap.add_argument("--p_init", type=float, default=defaults.p_init, help="Initial Erdos–Renyi edge probability p (used if mean_degree <= 0).")
    ap.add_argument("--mean_degree", type=float, default=defaults.mean_degree_init, help="Target initial mean degree; if >0, sets p = mean_degree/(N-1).")

    # Event / graph dynamics
    ap.add_argument("--deltaK", type=int, default=defaults.deltaK, help="Max absolute degree change per node-jump (Δk ∈ [-deltaK,deltaK]).")
    ap.add_argument("--xi_d", type=float, default=defaults.xi_d, help="DeepPER tunneling distance scale ξ_d.")
    ap.add_argument("--lambda0_edge", type=float, default=defaults.lambda0_edge, help="Global edge-event intensity scale λ0_edge.")
    ap.add_argument("--dt", type=float, default=defaults.dt, help="Environment step size Δt.")
    ap.add_argument("--rk4_steps", type=int, default=defaults.rk4_steps, help="RK4 sub-steps per environment step.")
    ap.add_argument("--discount_rate", type=float, default=defaults.discount_rate, help="Continuous-time discount rate γ in exp(-γ t).")
    ap.add_argument("--max_events_per_step", type=int, default=defaults.max_events_per_step, help="Cap on number of events processed per step.")
    ap.add_argument("--nonedge_candidates", type=int, default=defaults.nonedge_candidates, help="Non-edge candidates for creation jumps (linear-time cap).")

    # Edge recovery / stochastic reconnection (restore a fraction of removed edges per step)
    ap.add_argument("--edge_restore_low", type=float, default=defaults.edge_restore_frac_low, help="Lower bound for per-step edge restoration fraction (of removed-edge pool). Set to 0 to disable.")
    ap.add_argument("--edge_restore_high", type=float, default=defaults.edge_restore_frac_high, help="Upper bound for per-step edge restoration fraction (of removed-edge pool). Set to 0 to disable.")
    ap.add_argument("--edge_restore_max_per_step", type=int, default=defaults.edge_restore_max_per_step, help="If >0, cap the number of restored edges per step (safety/perf).")

    # Optimization
    ap.add_argument("--lr", type=float, default=defaults.lr, help="Learning rate.")
    ap.add_argument("--grad_clip", type=float, default=defaults.grad_clip, help="Gradient clipping norm.")

    # Runtime
    ap.add_argument("--seed", type=int, default=defaults.seed, help="Random seed.")
    ap.add_argument("--device", type=str, default=defaults.device, help="Torch device string: cpu | cuda | mps.")
    ap.add_argument("--deepper_exact", action="store_true", help="Compute exact DeepPER (small N only).")

    args = ap.parse_args()

    # PyCharm-friendly default: if you press Run without parameters, tests run.
    if (not args.run_tests) and (not args.demo):
        args.run_tests = True

    if args.run_tests:
        run_unit_tests()
        return

    if args.demo:
        # Build config from CLI overrides
        cfg = ACCAConfig(
            N=args.N,
            F=args.F,
            H=args.H,
            gcn_hidden=args.gcn_hidden,
            gcn_layers=args.gcn_layers,
            # Initial graph
            p_init=args.p_init,
            mean_degree_init=args.mean_degree,
            # Event / graph dynamics
            deltaK=args.deltaK,
            xi_d=args.xi_d,
            lambda0_edge=args.lambda0_edge,
            dt=args.dt,
            rk4_steps=args.rk4_steps,
            discount_rate=args.discount_rate,
            max_events_per_step=args.max_events_per_step,
            nonedge_candidates=args.nonedge_candidates,
            edge_restore_frac_low=args.edge_restore_low,
            edge_restore_frac_high=args.edge_restore_high,
            edge_restore_max_per_step=args.edge_restore_max_per_step,
            # Optimization
            lr=args.lr,
            grad_clip=args.grad_clip,
            # Runtime
            device=args.device,
            deepper_exact=bool(args.deepper_exact),
            seed=args.seed,
        )
        agent, env, history = train_accc(cfg, steps=args.steps, print_every=20, collect_history=True)

        # -------------------- "Accuracy" / performance summary --------------------
        summ = summarize_history(history, window=min(50, len(history)) if history else 1)
        print("\n=== Training summary (last window) ===")
        for k, v in summ.items():
            print(f"{k:>14s}: {v: .6f}")

        # State-prediction "accuracy proxy"
        if "avg_state_cos" in summ:
            print(f"\nState prediction cosine (accuracy proxy): {summ['avg_state_cos']:.4f} (1.0 is best)")

        # Event prediction proxies
        if "avg_type_acc" in summ:
            print(f"Event type accuracy (proxy): {summ['avg_type_acc']:.4f} (1.0 is best)")
        if "avg_node_hit10" in summ:
            print(f"Node event hit@10 (proxy): {summ['avg_node_hit10']:.4f}")
        if "avg_edge_hit10" in summ:
            print(f"Edge event hit@10 (proxy): {summ['avg_edge_hit10']:.4f}")

        # -------------------- Quick evaluation vs random baseline --------------------
        print("\n=== Evaluation (fresh env) ===")
        eval_mode = evaluate_policy(agent, cfg, steps=args.eval_steps, policy="mode", seed=cfg.seed + 1000)
        eval_rand = evaluate_policy(agent, cfg, steps=args.eval_steps, policy="random", seed=cfg.seed + 2000)
        print(f"Policy=mode   avg_reward_int={eval_mode['avg_reward_int']:.4f} avg_rssi={eval_mode['avg_rssi']:.2f} avg_zeta={eval_mode['avg_zeta']:.4f} avg_gcc={eval_mode['avg_gcc']:.4f} type_acc={eval_mode.get('avg_type_acc', float('nan')):.3f} node_hit10={eval_mode.get('avg_node_hit10', float('nan')):.3f} edge_hit10={eval_mode.get('avg_edge_hit10', float('nan')):.3f}")
        print(f"Policy=random avg_reward_int={eval_rand['avg_reward_int']:.4f} avg_rssi={eval_rand['avg_rssi']:.2f} avg_zeta={eval_rand['avg_zeta']:.4f} avg_gcc={eval_rand['avg_gcc']:.4f} type_acc={eval_rand.get('avg_type_acc', float('nan')):.3f} node_hit10={eval_rand.get('avg_node_hit10', float('nan')):.3f} edge_hit10={eval_rand.get('avg_edge_hit10', float('nan')):.3f}")

        # -------------------- Visualization --------------------
        if args.plot:
            import os
            import time as _time
            run_dir = os.path.join(args.outdir, _time.strftime("run_%Y%m%d_%H%M%S"))
            os.makedirs(run_dir, exist_ok=True)

            plot_history(history, run_dir, show=args.show_plots)
            plot_graph_snapshot(env.graph, env.pos, os.path.join(run_dir, "graph_final.png"), title="Final graph snapshot", show=args.show_plots)

        return


if __name__ == "__main__":
    main()
